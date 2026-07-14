#pragma once
#include <vector>
#include <thread>
#include <atomic>
#include <coroutine>
#include <cstdint>
#include <cstddef>
#include <climits>
#include <cassert>
#include <memory>
#include <mutex>
#include <new>
#include <semaphore>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <blockingconcurrentqueue.h>

namespace ic
{

    class JobSystem;

    // Intrusive node used to park a suspended coroutine on a JobCounter.
    //
    // The node lives *inside the awaiter object*, which in turn lives inside
    // the suspended coroutine's frame — never on a transient stack frame.
    // That is what makes it safe for another thread's decrement() to read the
    // node after the await_suspend has returned: the frame stays alive for the
    // entire duration of the suspension (see CounterAwaiter in task.h).
    struct CounterWaitNode
    {
        std::coroutine_handle<> handle {};
        JobSystem*              system { nullptr };
        CounterWaitNode*        next   { nullptr };
    };

    /*

        Job System — executes frame DAG nodes and graph passes in parallel.

        Scheduling invariants:
          - Each worker owns the bottom of one fixed Chase-Lev deque. It runs
            local work LIFO; thieves claim the top FIFO. Slot claim states make
            moving a JobTask safe without deque resizing or epoch reclamation.
          - External work and bounded-deque overflow use one global blocking
            injection queue. Workers periodically service it to prevent
            starvation even under a recursive local workload.
          - Workers spin briefly, then park using a generation/semaphore
            handshake. A publisher either changes the observed generation or
            sees the announced sleeper, so wakeups cannot be lost.
          - Every queued, executing, or counter-suspended coroutine owns one
            outstanding unit. Draining ends only when that total reaches zero.
          - Running accepts external work. Draining rejects new external work
            but accepts internal children/continuations. Stopped owns no work.

    */

    class JobCounter
    {
    public:
        JobCounter() = default;

        void increment(uint32_t count = 1)
        {
            if (count == 0) return;
            if (count > kCountMask)
            {
                throw std::overflow_error("JobCounter increment overflow");
            }

            uint32_t state = m_state.load(std::memory_order_acquire);
            for (;;)
            {
                if (state == kTransition)
                {
                    m_state.wait(state, std::memory_order_acquire);
                    state = m_state.load(std::memory_order_acquire);
                    continue;
                }

                if (state == 0)
                {
                    if (!m_state.compare_exchange_weak(
                            state, kTransition,
                            std::memory_order_acq_rel,
                            std::memory_order_acquire))
                    {
                        continue;
                    }

                    // Publish an open waiter list before publishing the
                    // positive count. pushWaiter uses the same short lock, so
                    // it can never observe work with a stale closed list.
                    lockWaiters();
                    m_waiters = nullptr;
                    m_state.store(count, std::memory_order_release);
                    unlockWaiters();
                    m_state.notify_all();
                    return;
                }

                if (state > kCountMask - count)
                {
                    throw std::overflow_error("JobCounter increment overflow");
                }
                if (m_state.compare_exchange_weak(
                        state, state + count,
                        std::memory_order_release,
                        std::memory_order_acquire))
                {
                    return;
                }
            }
        }

        void decrement(uint32_t count = 1)
        {
            if (count == 0) return;

            uint32_t state = m_state.load(std::memory_order_acquire);
            for (;;)
            {
                if (state == kTransition)
                {
                    m_state.wait(state, std::memory_order_acquire);
                    state = m_state.load(std::memory_order_acquire);
                    continue;
                }

                assert(state >= count);
                if (state < count) return;

                const uint32_t next = state == count
                    ? kTransition : state - count;
                if (!m_state.compare_exchange_weak(
                        state, next,
                        std::memory_order_acq_rel,
                        std::memory_order_acquire))
                {
                    continue;
                }

                if (next == kTransition)
                {
                    notifyWaiters();
                }
                return;
            }
        }

        uint32_t value() const
        {
            return m_state.load(std::memory_order_acquire) & kCountMask;
        }

        bool done() const
        {
            return m_state.load(std::memory_order_acquire) == 0;
        }

        void wait() const noexcept
        {
            uint32_t state = m_state.load(std::memory_order_acquire);
            while (state != 0)
            {
                m_state.wait(state, std::memory_order_acquire);
                state = m_state.load(std::memory_order_acquire);
            }
        }

        // Coroutine bridge
        //
        // Intrusive list of parked coroutines. Registration and the rare
        // zero/rearm transitions share a very short atomic-flag lock; ordinary
        // decrements remain lock-free. Coupling the list transition to the
        // count's kTransition state prevents both lost wakeups and reuse ABA.
        //
        // Returns true if the node was published (caller stays suspended), false
        // if the counter had already reached zero and closed the list (caller
        // must resume inline; the node was NOT published, so it is safe to keep
        // using the awaiter/frame).
        bool pushWaiter(CounterWaitNode* node)
        {
            for (;;)
            {
                lockWaiters();
                const uint32_t state = m_state.load(std::memory_order_acquire);
                if (state == kTransition)
                {
                    unlockWaiters();
                    m_state.wait(state, std::memory_order_acquire);
                    continue;
                }
                if (state == 0)
                {
                    unlockWaiters();
                    return false;
                }

                node->next = m_waiters;
                m_waiters = node;
                unlockWaiters();
                return true;
            }
        }

        // Sentinel head meaning "counter already hit zero; no further parking".
        static CounterWaitNode* closedSentinel() noexcept
        {
            return reinterpret_cast<CounterWaitNode*>(static_cast<std::uintptr_t>(1));
        }

    private:
        // Out of line: only reached on the rare zero transition with waiters
        // path, so it stays out of the hot decrement inline body.
        void notifyWaiters();

        void lockWaiters() const noexcept
        {
            while (m_waiterLock.test_and_set(std::memory_order_acquire))
            {
                m_waiterLock.wait(true, std::memory_order_relaxed);
            }
        }

        void unlockWaiters() const noexcept
        {
            m_waiterLock.clear(std::memory_order_release);
            m_waiterLock.notify_one();
        }

        static constexpr uint32_t kTransition = uint32_t{1} << 31u;
        static constexpr uint32_t kCountMask = kTransition - 1u;

        std::atomic<uint32_t> m_state { 0 };
        mutable std::atomic_flag m_waiterLock = ATOMIC_FLAG_INIT;
        CounterWaitNode* m_waiters { closedSentinel() };
    };



    // Move-only callable with a cache-friendly inline fast path. Renderer jobs
    // overwhelmingly capture a handful of pointers/indices, so ordinary
    // submission performs no allocation. Oversized or over-aligned callables
    // safely fall back to one heap allocation.
    class SmallJobFunction
    {
    public:
        static constexpr std::size_t InlineBytes = 48;

        SmallJobFunction() noexcept = default;
        SmallJobFunction(const SmallJobFunction&) = delete;
        SmallJobFunction& operator=(const SmallJobFunction&) = delete;

        SmallJobFunction(SmallJobFunction&& other) noexcept
        {
            moveFrom(std::move(other));
        }

        SmallJobFunction& operator=(SmallJobFunction&& other) noexcept
        {
            if (this != &other)
            {
                reset();
                moveFrom(std::move(other));
            }
            return *this;
        }

        ~SmallJobFunction() { reset(); }

        template<typename F>
            requires (!std::is_same_v<std::remove_cvref_t<F>, SmallJobFunction>)
        explicit SmallJobFunction(F&& function)
        {
            using Fn = std::remove_cvref_t<F>;
            static_assert(std::is_invocable_r_v<void, Fn&>,
                "Job callables must be invocable as void()");

            if constexpr (fitsInline<Fn>)
            {
                ::new (static_cast<void*>(m_storage)) Fn(std::forward<F>(function));
                m_ops = &inlineOps<Fn>();
            }
            else
            {
                Fn* allocation = new Fn(std::forward<F>(function));
                ::new (static_cast<void*>(m_storage)) Fn*(allocation);
                m_ops = &heapOps<Fn>();
            }
        }

        explicit operator bool() const noexcept { return m_ops != nullptr; }
        void operator()() { m_ops->invoke(m_storage); }

    private:
        struct Ops
        {
            void (*invoke)(void*);
            void (*move)(void*, void*) noexcept;
            void (*destroy)(void*) noexcept;
        };

        template<typename Fn>
        static constexpr bool fitsInline =
            sizeof(Fn) <= InlineBytes &&
            alignof(Fn) <= alignof(std::max_align_t) &&
            std::is_nothrow_move_constructible_v<Fn>;

        template<typename Fn>
        static const Ops& inlineOps() noexcept
        {
            static const Ops ops {
                [](void* storage) { (*std::launder(reinterpret_cast<Fn*>(storage)))(); },
                [](void* source, void* destination) noexcept
                {
                    Fn* fn = std::launder(reinterpret_cast<Fn*>(source));
                    ::new (destination) Fn(std::move(*fn));
                    std::destroy_at(fn);
                },
                [](void* storage) noexcept
                {
                    std::destroy_at(std::launder(reinterpret_cast<Fn*>(storage)));
                }
            };
            return ops;
        }

        template<typename Fn>
        static const Ops& heapOps() noexcept
        {
            static const Ops ops {
                [](void* storage)
                {
                    Fn* fn = *std::launder(reinterpret_cast<Fn**>(storage));
                    (*fn)();
                },
                [](void* source, void* destination) noexcept
                {
                    Fn*& fn = *std::launder(reinterpret_cast<Fn**>(source));
                    ::new (destination) Fn*(std::exchange(fn, nullptr));
                    std::destroy_at(std::launder(reinterpret_cast<Fn**>(source)));
                },
                [](void* storage) noexcept
                {
                    Fn* fn = *std::launder(reinterpret_cast<Fn**>(storage));
                    delete fn;
                    std::destroy_at(std::launder(reinterpret_cast<Fn**>(storage)));
                }
            };
            return ops;
        }

        void reset() noexcept
        {
            if (m_ops)
            {
                m_ops->destroy(m_storage);
                m_ops = nullptr;
            }
        }

        void moveFrom(SmallJobFunction&& other) noexcept
        {
            m_ops = std::exchange(other.m_ops, nullptr);
            if (m_ops) m_ops->move(other.m_storage, m_storage);
        }

        alignas(std::max_align_t) std::byte m_storage[InlineBytes] {};
        const Ops* m_ops { nullptr };
    };

    struct JobTask
    {
        SmallJobFunction function;
        std::coroutine_handle<> coroutine {};
        JobCounter* counter { nullptr };

        JobTask() noexcept = default;
        JobTask(JobTask&&) noexcept = default;
        JobTask& operator=(JobTask&&) noexcept = default;
        JobTask(const JobTask&) = delete;
        JobTask& operator=(const JobTask&) = delete;

        template<typename F>
        static JobTask make(F&& fn, JobCounter* ctr = nullptr)
        {
            JobTask t;
            t.counter = ctr;
            t.function = SmallJobFunction(std::forward<F>(fn));
            return t;
        }
    };

    class JobSystem
    {
    public:
        enum class Lifecycle : uint8_t
        {
            Running,
            Draining,
            Stopped
        };

        JobSystem();
        ~JobSystem();
        JobSystem(const JobSystem&) = delete;
        JobSystem& operator=(const JobSystem&) = delete;

        void init(uint32_t threadCount = 0);
        void shutdown();

        // Consume an array of move-only tasks. Worker submissions prefer that
        // worker's local deque; external submissions and local overflow enter
        // the injection queue. The optional counter is retired automatically.
                 
        void kickTasks(
            JobTask* tasks,
            uint32_t count,
            JobCounter* counter = nullptr);

        // Blocks the calling thread (main thread)
        // and works on remaining jobs until counter is 0
        void waitForCounter(JobCounter* counter);

        uint32_t workerCount() const
        {
            return m_workerCount.load(std::memory_order_acquire);
        }

        // Coroutine integration
        //
        // Schedule a direct coroutine payload, preferring the current worker's
        // deque. Continuations are internal work and remain legal while draining.
        void scheduleResume(std::coroutine_handle<> handle);

        // Run at most one queued job on the calling thread if one is available.
        // Returns true if a job was executed. Used by sync_wait() to let a
        // non worker (e.g. the main thread) help drain the queue while it blocks
        // at the coroutine/blocking boundary, mirroring waitForCounter().
        bool tryRunOne();

        Lifecycle lifecycle() const noexcept
        {
            return m_lifecycle.load(std::memory_order_acquire);
        }

        // Scheduler-internal lifetime accounting used by CounterAwaiter. A
        // parked coroutine is outstanding work even though it is not queued.
        void retainSuspended() noexcept;
        void releaseSuspended() noexcept;

    private:

        struct WorkerState;

        // Runs task.function, then decrements task.counter if present.
        // Single authoritative call site for task completion
        void executeTask(JobTask& task);

        void workerLoop(uint32_t workerIndex);
        bool tryAcquireTask(JobTask& task, WorkerState* worker);
        bool trySteal(JobTask& task, WorkerState* thief);
        bool submitOne(JobTask&& task, WorkerState* localWorker);
        void publishWork(uint32_t count = 1) noexcept;
        void finishOutstanding() noexcept;
        WorkerState* currentWorker() noexcept;

        std::vector<std::unique_ptr<WorkerState>> m_workerStates;
        std::vector<std::thread> m_workers;
        moodycamel::BlockingConcurrentQueue<JobTask> m_injectionQueue;
        std::counting_semaphore<INT_MAX> m_workSemaphore { 0 };
        std::atomic<Lifecycle> m_lifecycle { Lifecycle::Running };
        std::atomic<uint64_t> m_outstanding { 0 };
        std::atomic<uint64_t> m_wakeGeneration { 0 };
        std::atomic<uint32_t> m_sleepers { 0 };
        std::atomic<uint32_t> m_externalSubmitters { 0 };
        std::atomic<uint32_t> m_activeHelpers { 0 };
        std::atomic<uint32_t> m_workerCount { 0 };
        std::atomic<bool> m_initialized { false };
        std::mutex m_lifecycleMutex;
    };
}

