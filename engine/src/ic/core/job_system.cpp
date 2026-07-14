#include "ic/common/ic_pch.h"
#include "ic/core/job_system.h"
#include "ic/util/profiler.h"
#include <spdlog/spdlog.h>

#include <array>
#include <chrono>
#include <limits>

namespace ic
{
    namespace
    {
        // 4096 entries covers large frame-graph/coroutine fan-outs while
        // remaining a fixed, bounded allocation (~320 KiB per worker). Larger
        // bursts overflow safely into the global injection queue.
        constexpr uint32_t kLocalDequeCapacity = 4096;
        constexpr uint32_t kExternalFairnessInterval = 32;
        constexpr uint32_t kSpinCount = 128;

        static_assert((kLocalDequeCapacity & (kLocalDequeCapacity - 1)) == 0,
            "The Chase-Lev capacity must be a power of two");

        // TLS contains no owning pointer and is cleared before a worker exits.
        // Matching both the scheduler and worker pointer prevents an identity
        // from a previous JobSystem lifetime being reused accidentally.
        struct WorkerTls
        {
            JobSystem* system { nullptr };
            void* worker { nullptr };
            JobSystem* executingSystem { nullptr };
        };

        thread_local WorkerTls g_workerTls;

        inline void spinPause() noexcept
        {
#if defined(_WIN32)
            YieldProcessor();
#elif defined(__i386__) || defined(__x86_64__)
            __builtin_ia32_pause();
#else
            std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
        }

        // Fixed-capacity Chase-Lev deque. Only the owning worker mutates the
        // bottom; thieves contend on top. Slots have an explicit claim state so
        // move-only JobTasks remain safe after a thief wins top: the owner may
        // observe the logical space immediately, but cannot reuse that physical
        // slot until the winner finishes moving the payload out.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4324) // intentional cache-line isolation
#endif
        class ChaseLevDeque
        {
        public:
            bool push(JobTask&& task) noexcept
            {
                const uint64_t bottom = m_bottom.load(std::memory_order_relaxed);
                const uint64_t top = m_top.load(std::memory_order_acquire);
                if (bottom - top >= kLocalDequeCapacity) return false;

                Slot& slot = m_slots[bottom & (kLocalDequeCapacity - 1)];
                if (slot.state.load(std::memory_order_acquire) != SlotState::Empty)
                {
                    return false;
                }

                slot.task = std::move(task);
                slot.state.store(SlotState::Ready, std::memory_order_release);
                m_bottom.store(bottom + 1, std::memory_order_release);
                return true;
            }

            bool pop(JobTask& output) noexcept
            {
                uint64_t bottom = m_bottom.load(std::memory_order_relaxed);
                if (bottom == 0) return false;

                --bottom;
                m_bottom.store(bottom, std::memory_order_relaxed);
                std::atomic_thread_fence(std::memory_order_seq_cst);
                uint64_t top = m_top.load(std::memory_order_relaxed);

                if (top > bottom)
                {
                    m_bottom.store(bottom + 1, std::memory_order_relaxed);
                    return false;
                }

                if (top == bottom)
                {
                    if (!m_top.compare_exchange_strong(
                            top, top + 1,
                            std::memory_order_seq_cst,
                            std::memory_order_relaxed))
                    {
                        m_bottom.store(bottom + 1, std::memory_order_relaxed);
                        return false;
                    }
                    m_bottom.store(bottom + 1, std::memory_order_relaxed);
                }

                takeClaimedSlot(bottom, output);
                return true;
            }

            bool steal(JobTask& output) noexcept
            {
                uint64_t top = m_top.load(std::memory_order_acquire);
                std::atomic_thread_fence(std::memory_order_seq_cst);
                const uint64_t bottom = m_bottom.load(std::memory_order_acquire);
                if (top >= bottom) return false;

                if (!m_top.compare_exchange_strong(
                        top, top + 1,
                        std::memory_order_seq_cst,
                        std::memory_order_relaxed))
                {
                    return false;
                }

                takeClaimedSlot(top, output);
                return true;
            }

            bool empty() const noexcept
            {
                return m_top.load(std::memory_order_acquire) >=
                    m_bottom.load(std::memory_order_acquire);
            }

        private:
            enum class SlotState : uint8_t { Empty, Ready, Claimed };

            struct Slot
            {
                std::atomic<SlotState> state { SlotState::Empty };
                JobTask task;
            };

            void takeClaimedSlot(uint64_t index, JobTask& output) noexcept
            {
                Slot& slot = m_slots[index & (kLocalDequeCapacity - 1)];
                // The bottom/top protocol has already awarded this logical
                // index to exactly one consumer. State protects physical slot
                // reuse for move-only payloads; it does not need another
                // contended ownership CAS here.
                while (slot.state.load(std::memory_order_acquire) !=
                    SlotState::Ready)
                {
                    spinPause();
                }
                slot.state.store(SlotState::Claimed, std::memory_order_relaxed);
                output = std::move(slot.task);
                slot.state.store(SlotState::Empty, std::memory_order_release);
                slot.state.notify_one();
            }

            alignas(64) std::atomic<uint64_t> m_top { 0 };
            alignas(64) std::atomic<uint64_t> m_bottom { 0 };
            std::array<Slot, kLocalDequeCapacity> m_slots;
        };
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    }

    struct alignas(64) JobSystem::WorkerState
    {
        explicit WorkerState(uint32_t workerIndex) noexcept
            : index(workerIndex), randomState(0x9E3779B97F4A7C15ull ^
                (static_cast<uint64_t>(workerIndex) + 1) * 0xBF58476D1CE4E5B9ull)
        {
        }

        ChaseLevDeque deque;
        uint32_t index { 0 };
        uint32_t localRun { 0 };
        uint64_t randomState { 0 };
    };

    JobSystem::JobSystem() = default;

    JobSystem::~JobSystem()
    {
        shutdown();
    }

    void JobSystem::init(uint32_t threadCount)
    {
        std::scoped_lock lifecycleLock(m_lifecycleMutex);
        bool expected = false;
        if (!m_initialized.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel))
        {
            return;
        }

        if (m_outstanding.load(std::memory_order_acquire) != 0)
        {
            m_initialized.store(false, std::memory_order_release);
            throw std::logic_error("JobSystem::init called with outstanding work");
        }

        m_lifecycle.store(Lifecycle::Running, std::memory_order_release);
        if (threadCount == 0)
        {
            const uint32_t hardwareThreads = std::thread::hardware_concurrency();
            threadCount = hardwareThreads > 1 ? hardwareThreads - 1 : 1;
        }

        spdlog::info("[JobSystem] Starting {} worker threads", threadCount);
        m_workerStates.reserve(threadCount);
        m_workers.reserve(threadCount);
        for (uint32_t i = 0; i < threadCount; ++i)
        {
            m_workerStates.push_back(std::make_unique<WorkerState>(i));
        }
        m_workerCount.store(threadCount, std::memory_order_release);
        for (uint32_t i = 0; i < threadCount; ++i)
        {
            m_workers.emplace_back(&JobSystem::workerLoop, this, i);

#if defined(_WIN32)
            auto handle = m_workers.back().native_handle();
            const std::wstring name = L"JobWorker-" + std::to_wstring(i);
            SetThreadDescription(handle, name.c_str());
#elif defined(__linux__)
            auto handle = m_workers.back().native_handle();
            const std::string name = "JobWorker-" + std::to_string(i);
            pthread_setname_np(handle, name.c_str());
#endif
            spdlog::info("[JobSystem]   JobWorker-{} started", i);
        }
    }

    void JobSystem::shutdown()
    {
        if (currentWorker())
        {
            throw std::logic_error(
                "JobSystem::shutdown cannot block from one of its own workers");
        }
        std::scoped_lock lifecycleLock(m_lifecycleMutex);
        Lifecycle expected = Lifecycle::Running;
        if (!m_lifecycle.compare_exchange_strong(
                expected, Lifecycle::Draining,
                std::memory_order_acq_rel,
                std::memory_order_acquire))
        {
            if (expected == Lifecycle::Stopped) return;
            return;
        }

        // Closing Running first makes all later external entrants reject. An
        // entrant already inside the gate must publish its outstanding units
        // (or roll them back) before draining is allowed to complete.
        uint32_t submitters = m_externalSubmitters.load(std::memory_order_acquire);
        while (submitters != 0)
        {
            m_externalSubmitters.wait(submitters, std::memory_order_acquire);
            submitters = m_externalSubmitters.load(std::memory_order_acquire);
        }

        publishWork(std::numeric_limits<uint32_t>::max());

        // The shutdown caller participates. This is essential for the valid
        // zero-worker mode and reduces tail latency when only a few jobs remain.
        while (m_outstanding.load(std::memory_order_acquire) != 0)
        {
            if (!tryRunOne())
            {
                std::this_thread::yield();
            }
        }

        for (std::thread& thread : m_workers)
        {
            if (thread.joinable()) thread.join();
        }
        m_workers.clear();

        // Stop new public helper calls, then wait for any helper that began
        // during Draining to leave the steal path before deque storage is freed.
        m_lifecycle.store(Lifecycle::Stopped, std::memory_order_release);
        uint32_t helpers = m_activeHelpers.load(std::memory_order_acquire);
        while (helpers != 0)
        {
            m_activeHelpers.wait(helpers, std::memory_order_acquire);
            helpers = m_activeHelpers.load(std::memory_order_acquire);
        }
        m_workerStates.clear();
        m_workerCount.store(0, std::memory_order_release);
        m_initialized.store(false, std::memory_order_release);
        spdlog::info("[JobSystem] All workers joined");
    }

    JobSystem::WorkerState* JobSystem::currentWorker() noexcept
    {
        if (g_workerTls.system != this) return nullptr;
        return static_cast<WorkerState*>(g_workerTls.worker);
    }

    bool JobSystem::submitOne(JobTask&& task, WorkerState* localWorker)
    {
        if (localWorker && localWorker->deque.push(std::move(task)))
        {
            return true;
        }
        return m_injectionQueue.enqueue(std::move(task));
    }

    void JobSystem::kickTasks(JobTask* tasks, uint32_t count, JobCounter* counter)
    {
        if (count == 0) return;
        if (!tasks)
        {
            throw std::invalid_argument("JobSystem::kickTasks received null tasks");
        }

        WorkerState* localWorker = currentWorker();
        // A helping caller has no local deque, but work it is currently
        // executing is still internal and must remain legal during Draining.
        const bool external = localWorker == nullptr &&
            g_workerTls.executingSystem != this;
        if (external)
        {
            m_externalSubmitters.fetch_add(1, std::memory_order_acq_rel);
            if (m_lifecycle.load(std::memory_order_acquire) != Lifecycle::Running)
            {
                if (m_externalSubmitters.fetch_sub(1, std::memory_order_acq_rel) == 1)
                {
                    m_externalSubmitters.notify_all();
                }
                throw std::runtime_error("JobSystem is draining; external work rejected");
            }
        }
        else if (m_lifecycle.load(std::memory_order_acquire) == Lifecycle::Stopped)
        {
            throw std::runtime_error("JobSystem is stopped");
        }

        if (counter) counter->increment(count);
        m_outstanding.fetch_add(count, std::memory_order_acq_rel);

        uint32_t queued = 0;
        try
        {
            for (; queued < count; ++queued)
            {
                tasks[queued].counter = counter;
                if (!submitOne(std::move(tasks[queued]), localWorker))
                {
                    throw std::bad_alloc{};
                }
            }
        }
        catch (...)
        {
            const uint32_t unqueued = count - queued;
            if (counter) counter->decrement(unqueued);
            if (m_outstanding.fetch_sub(unqueued, std::memory_order_acq_rel) == unqueued)
            {
                m_outstanding.notify_all();
            }
            if (external &&
                m_externalSubmitters.fetch_sub(1, std::memory_order_acq_rel) == 1)
            {
                m_externalSubmitters.notify_all();
            }
            publishWork(queued);
            throw;
        }

        if (external &&
            m_externalSubmitters.fetch_sub(1, std::memory_order_acq_rel) == 1)
        {
            m_externalSubmitters.notify_all();
        }
        publishWork(count);
    }

    void JobSystem::scheduleResume(std::coroutine_handle<> handle)
    {
        ZoneScopedN("JobSystem::scheduleResume");
        if (!handle) return;
        if (m_lifecycle.load(std::memory_order_acquire) == Lifecycle::Stopped)
        {
            std::terminate();
        }

        JobTask task;
        task.coroutine = handle;
        m_outstanding.fetch_add(1, std::memory_order_acq_rel);
        if (!submitOne(std::move(task), currentWorker()))
        {
            finishOutstanding();
            std::terminate();
        }
        publishWork();
    }

    void JobSystem::retainSuspended() noexcept
    {
        if (m_lifecycle.load(std::memory_order_acquire) == Lifecycle::Stopped)
        {
            std::terminate();
        }
        m_outstanding.fetch_add(1, std::memory_order_acq_rel);
    }

    void JobSystem::releaseSuspended() noexcept
    {
        finishOutstanding();
    }

    void JobSystem::finishOutstanding() noexcept
    {
        const uint64_t previous = m_outstanding.fetch_sub(1, std::memory_order_acq_rel);
        assert(previous != 0);
        if (previous == 1)
        {
            m_outstanding.notify_all();
            if (m_lifecycle.load(std::memory_order_acquire) == Lifecycle::Draining)
            {
                publishWork(std::numeric_limits<uint32_t>::max());
            }
        }
    }

    void JobSystem::publishWork(uint32_t count) noexcept
    {
        // Publisher half of the lost-wakeup handshake (worker half is in
        // workerLoop's park section). Correctness requires that the generation
        // bump be globally ordered before the sleeper load, mirrored on the
        // worker side by the sleeper increment being ordered before its
        // generation recheck. Both sides use a read-modify-write followed by an
        // acquire load; on x86-64 (the only build target) the RMW is a full
        // barrier, so the StoreLoad cannot reorder and the two sides cannot both
        // miss each other. A weak-memory port would have to strengthen these
        // fetch_add/load pairs to seq_cst (or add a seq_cst fence between them).
        m_wakeGeneration.fetch_add(1, std::memory_order_release);
        const uint32_t sleepers = m_sleepers.load(std::memory_order_acquire);
        const uint32_t wakeCount = std::min(count, sleepers);
        if (wakeCount != 0)
        {
            m_workSemaphore.release(static_cast<std::ptrdiff_t>(wakeCount));
        }
    }

    bool JobSystem::trySteal(JobTask& task, WorkerState* thief)
    {
        const uint32_t workerCount = static_cast<uint32_t>(m_workerStates.size());
        if (workerCount == 0) return false;

        uint64_t random = thief ? thief->randomState :
            (m_wakeGeneration.load(std::memory_order_relaxed) |
                0xD1B54A32D192ED03ull);
        random ^= random >> 12;
        random ^= random << 25;
        random ^= random >> 27;
        if (thief) thief->randomState = random;

        const uint32_t start = static_cast<uint32_t>(random % workerCount);
        for (uint32_t offset = 0; offset < workerCount; ++offset)
        {
            WorkerState* victim = m_workerStates[(start + offset) % workerCount].get();
            if (victim == thief) continue;
            if (victim->deque.steal(task)) return true;
        }
        return false;
    }

    bool JobSystem::tryAcquireTask(JobTask& task, WorkerState* worker)
    {
        if (worker && worker->localRun < kExternalFairnessInterval &&
            worker->deque.pop(task))
        {
            ++worker->localRun;
            return true;
        }

        if (m_injectionQueue.try_dequeue(task))
        {
            if (worker) worker->localRun = 0;
            return true;
        }

        if (worker && worker->deque.pop(task))
        {
            worker->localRun = 1;
            return true;
        }

        if (trySteal(task, worker))
        {
            if (worker) worker->localRun = 0;
            return true;
        }
        return false;
    }

    bool JobSystem::tryRunOne()
    {
        WorkerState* worker = currentWorker();
        const bool helper = worker == nullptr;
        if (helper)
        {
            m_activeHelpers.fetch_add(1, std::memory_order_acq_rel);
            if (m_lifecycle.load(std::memory_order_acquire) == Lifecycle::Stopped)
            {
                if (m_activeHelpers.fetch_sub(1, std::memory_order_acq_rel) == 1)
                {
                    m_activeHelpers.notify_all();
                }
                return false;
            }
        }

        JobTask task;
        const bool acquired = tryAcquireTask(task, worker);
        if (acquired) executeTask(task);
        if (helper && m_activeHelpers.fetch_sub(1, std::memory_order_acq_rel) == 1)
        {
            m_activeHelpers.notify_all();
        }
        return acquired;
    }

    void JobSystem::waitForCounter(JobCounter* counter)
    {
        if (!counter || counter->done()) return;
        while (!counter->done())
        {
            if (!tryRunOne())
            {
                if (workerCount() == 0) std::this_thread::yield();
                else counter->wait();
            }
        }
    }

    void JobCounter::notifyWaiters()
    {
        lockWaiters();
        CounterWaitNode* node = m_waiters;
        m_waiters = closedSentinel();
        m_state.store(0, std::memory_order_release);
        unlockWaiters();
        m_state.notify_all();

        while (node && node != closedSentinel())
        {
            CounterWaitNode* next = node->next;
            JobSystem* system = node->system;
            const std::coroutine_handle<> handle = node->handle;
            if (system)
            {
                // Convert the parked lifetime unit into a queued lifetime unit
                // without ever allowing shutdown to observe a zero gap.
                system->scheduleResume(handle);
                system->releaseSuspended();
            }
            node = next;
        }
    }

    void JobSystem::executeTask(JobTask& task)
    {
        JobSystem* previousExecuting = g_workerTls.executingSystem;
        g_workerTls.executingSystem = this;
        try
        {
            if (task.coroutine) task.coroutine.resume();
            else if (task.function) task.function();
        }
        catch (const std::exception& error)
        {
            spdlog::error("[JobSystem] Unhandled job exception: {}", error.what());
        }
        catch (...)
        {
            spdlog::error("[JobSystem] Unhandled non-standard job exception");
        }
        g_workerTls.executingSystem = previousExecuting;

        if (task.counter) task.counter->decrement();
        // A worker reuses one JobTask object. Clear the consumed coroutine
        // handle before any generation-only wakeup can inspect that object;
        // resuming a stale final-suspended handle would be a use-after-free.
        task = JobTask{};
        finishOutstanding();
    }

    void JobSystem::workerLoop(uint32_t workerIndex)
    {
        WorkerState* worker = m_workerStates[workerIndex].get();
        g_workerTls = { this, worker, nullptr };

        JobTask task;
        for (;;)
        {
            if (tryAcquireTask(task, worker))
            {
                executeTask(task);
                continue;
            }

            bool found = false;
            for (uint32_t spin = 0; spin < kSpinCount; ++spin)
            {
                if (tryAcquireTask(task, worker))
                {
                    found = true;
                    break;
                }
                spinPause();
            }
            if (found)
            {
                executeTask(task);
                continue;
            }

            if (m_lifecycle.load(std::memory_order_acquire) != Lifecycle::Running &&
                m_outstanding.load(std::memory_order_acquire) == 0)
            {
                break;
            }

            // Lost-wakeup proof park: observe generation, announce sleeping,
            // then recheck both work and generation. A publisher either changes
            // generation before the recheck or sees this sleeper and releases a
            // semaphore token after it.
            const uint64_t observed =
                m_wakeGeneration.load(std::memory_order_acquire);
            m_sleepers.fetch_add(1, std::memory_order_acq_rel);

            if (tryAcquireTask(task, worker) ||
                m_wakeGeneration.load(std::memory_order_acquire) != observed)
            {
                m_sleepers.fetch_sub(1, std::memory_order_acq_rel);
                if (task.function || task.coroutine) executeTask(task);
                continue;
            }

            m_workSemaphore.acquire();
            m_sleepers.fetch_sub(1, std::memory_order_acq_rel);
        }

        g_workerTls = {};
    }
}
