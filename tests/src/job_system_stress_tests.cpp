#include "common/tests_pch.h"
#include "job_system_stress_tests.h"

#include "ic/core/job_system.h"
#include "ic/core/task.h"

#include <cstdlib>
#include <memory>
#include <numeric>
#include <span>
#include <type_traits>

namespace
{
    using namespace ic;

    [[noreturn]] void fail(const char* message)
    {
        std::cerr << "FAILED: " << message << '\n';
        std::abort();
    }

    void require(bool condition, const char* message)
    {
        if (!condition) fail(message);
    }

    Task<int> immediateValue(int value)
    {
        co_return value;
    }

    Task<void> immediateVoid(std::atomic<uint32_t>& count)
    {
        count.fetch_add(1, std::memory_order_relaxed);
        co_return;
    }

    Task<int> nestedValue(uint32_t depth)
    {
        if (depth == 0) co_return 1;
        co_return 1 + co_await nestedValue(depth - 1);
    }

    Task<void> throwingTask()
    {
        throw std::runtime_error("expected coroutine failure");
        co_return;
    }

    Task<std::unique_ptr<int>> moveOnlyValue(int value)
    {
        co_return std::make_unique<int>(value);
    }

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4324)
#endif
    struct alignas(64) OverAlignedValue
    {
        std::array<std::byte, 64> bytes {};
    };

    Task<OverAlignedValue> overAlignedValue()
    {
        co_return OverAlignedValue{};
    }
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

    Task<int> throwingValueTask()
    {
        throw std::runtime_error("expected when_all failure");
        co_return 0;
    }

    Task<int> countedValue(std::atomic<uint32_t>& completed, int value)
    {
        completed.fetch_add(1, std::memory_order_relaxed);
        co_return value;
    }

    Task<void> awaitJobs(
        JobSystem& jobs,
        std::span<JobTask> tasks,
        JobCounter& counter)
    {
        jobs.kickTasks(
            tasks.data(),
            static_cast<uint32_t>(tasks.size()),
            &counter);
        co_await awaitCounter(jobs, counter);
    }

    Task<void> awaitExistingCounter(
        JobSystem& jobs,
        JobCounter& counter,
        std::atomic<uint32_t>& resumed)
    {
        co_await awaitCounter(jobs, counter);
        resumed.fetch_add(1, std::memory_order_relaxed);
    }

    Task<void> drainingCoroutineChain(
        JobSystem& jobs,
        std::atomic<bool>& started,
        std::atomic<uint32_t>& completed)
    {
        constexpr uint32_t rounds = 128;
        constexpr uint32_t width = 32;
        for (uint32_t round = 0; round < rounds; ++round)
        {
            std::array<JobTask, width> tasks;
            for (JobTask& task : tasks)
            {
                task = JobTask::make([&completed]
                {
                    completed.fetch_add(1, std::memory_order_relaxed);
                });
            }
            JobCounter counter;
            jobs.kickTasks(tasks.data(), width, &counter);
            if (round == 0)
            {
                started.store(true, std::memory_order_release);
                started.notify_all();
            }
            co_await awaitCounter(jobs, counter);
        }
    }

    void stressRawJobs(JobSystem& jobs)
    {
        constexpr uint32_t taskCount = 100000;
        std::atomic<uint64_t> sum{ 0 };
        std::vector<JobTask> tasks;
        tasks.reserve(taskCount);
        for (uint32_t i = 1; i <= taskCount; ++i)
        {
            tasks.push_back(JobTask::make(
                [&, i] { sum.fetch_add(i, std::memory_order_relaxed); }));
        }

        JobCounter counter;
        jobs.kickTasks(tasks.data(), taskCount, &counter);
        jobs.waitForCounter(&counter);

        const uint64_t expected =
            static_cast<uint64_t>(taskCount) * (taskCount + 1) / 2;
        require(sum.load(std::memory_order_relaxed) == expected,
                "raw job fan-out lost or duplicated work");
    }

    void stressCounterReuse(JobSystem& jobs)
    {
        constexpr uint32_t rounds = 2000;
        constexpr uint32_t tasksPerRound = 32;
        std::atomic<uint32_t> completed{ 0 };
        JobCounter counter;
        for (uint32_t round = 0; round < rounds; ++round)
        {
            std::array<JobTask, tasksPerRound> tasks;
            for (JobTask& task : tasks)
            {
                task = JobTask::make([&]
                {
                    completed.fetch_add(1, std::memory_order_relaxed);
                });
            }
            sync_wait(jobs, awaitJobs(jobs, tasks, counter));
            require(counter.done(), "reused counter did not return to zero");
        }
        require(completed.load(std::memory_order_relaxed) ==
                    rounds * tasksPerRound,
                "counter reuse lost a completion");
    }

    void stressCoroutines(JobSystem& jobs)
    {
        require(sync_wait(jobs, nestedValue(4096)) == 4097,
                "deep symmetric-transfer chain returned the wrong value");

        constexpr uint32_t count = 10000;
        std::vector<Task<int>> values;
        values.reserve(count);
        for (uint32_t i = 0; i < count; ++i)
        {
            values.push_back(immediateValue(static_cast<int>(i)));
        }
        std::vector<int> results =
            sync_wait(jobs, when_all(jobs, std::move(values)));
        require(results.size() == count, "when_all returned the wrong size");
        require(std::accumulate(results.begin(), results.end(), uint64_t{ 0 }) ==
                    static_cast<uint64_t>(count) * (count - 1) / 2,
                "when_all corrupted result values");

        std::atomic<uint32_t> voidCount{ 0 };
        std::vector<Task<void>> voidTasks;
        voidTasks.reserve(count);
        for (uint32_t i = 0; i < count; ++i)
        {
            voidTasks.push_back(immediateVoid(voidCount));
        }
        sync_wait(jobs, when_all(jobs, std::move(voidTasks)));
        require(voidCount.load(std::memory_order_relaxed) == count,
                "void when_all lost work");

        bool caught = false;
        try
        {
            sync_wait(jobs, throwingTask());
        }
        catch (const std::runtime_error& error)
        {
            caught = std::string_view(error.what()) ==
                "expected coroutine failure";
        }
        require(caught, "coroutine exception did not propagate through sync_wait");

        std::unique_ptr<int> moveOnly = sync_wait(jobs, moveOnlyValue(73));
        require(moveOnly && *moveOnly == 73,
                "move-only coroutine result was not transferred correctly");

        bool emptyCaught = false;
        try
        {
            sync_wait(jobs, Task<int>{});
        }
        catch (const std::logic_error&)
        {
            emptyCaught = true;
        }
        require(emptyCaught, "awaiting an empty task did not report misuse");

        // Inspect the promise's actual result storage, not the aligned object
        // produced after sync_wait, to verify the coroutine allocation itself.
        Task<OverAlignedValue> alignedTask = overAlignedValue();
        alignedTask.handle().resume();
        OverAlignedValue&& alignedResult = alignedTask.handle().promise().result();
        require(reinterpret_cast<std::uintptr_t>(&alignedResult) %
                    alignof(OverAlignedValue) == 0,
                "over-aligned coroutine promise storage is misaligned");

        std::atomic<uint32_t> completedAfterFailure{ 0 };
        std::vector<Task<int>> mixedTasks;
        mixedTasks.push_back(countedValue(completedAfterFailure, 1));
        mixedTasks.push_back(throwingValueTask());
        mixedTasks.push_back(countedValue(completedAfterFailure, 2));
        caught = false;
        try
        {
            (void)sync_wait(jobs, when_all(jobs, std::move(mixedTasks)));
        }
        catch (const std::runtime_error&)
        {
            caught = true;
        }
        require(caught, "when_all did not propagate a child exception");
        require(completedAfterFailure.load(std::memory_order_relaxed) == 2,
                "when_all abandoned siblings after a child exception");
    }

    void stressConcurrentCounterEpisodes(JobSystem& jobs)
    {
        constexpr uint32_t rounds = 500;
        constexpr uint32_t waitersPerRound = 32;
        JobCounter counter;
        std::atomic<uint32_t> resumed{ 0 };

        for (uint32_t round = 0; round < rounds; ++round)
        {
            counter.increment();
            std::vector<Task<void>> waiters;
            waiters.reserve(waitersPerRound);
            for (uint32_t i = 0; i < waitersPerRound; ++i)
            {
                waiters.push_back(
                    awaitExistingCounter(jobs, counter, resumed));
            }

            JobTask release = JobTask::make(
                [&counter] { counter.decrement(); });
            jobs.kickTasks(&release, 1);
            sync_wait(jobs, when_all(jobs, std::move(waiters)));
            require(counter.done(),
                    "counter episode did not close before reuse");
        }

        require(resumed.load(std::memory_order_relaxed) ==
                    rounds * waitersPerRound,
                "rapid counter reuse lost or duplicated a waiter");
    }

    void stressSharedWaitersAndJobExceptions(JobSystem& jobs)
    {
        constexpr uint32_t waiterCount = 4096;
        JobCounter sharedCounter;
        sharedCounter.increment();
        std::atomic<uint32_t> resumed{ 0 };

        std::vector<Task<void>> waiters;
        waiters.reserve(waiterCount);
        for (uint32_t i = 0; i < waiterCount; ++i)
        {
            waiters.push_back(
                awaitExistingCounter(jobs, sharedCounter, resumed));
        }

        JobTask release = JobTask::make(
            [&sharedCounter] { sharedCounter.decrement(); });
        jobs.kickTasks(&release, 1);
        sync_wait(jobs, when_all(jobs, std::move(waiters)));
        require(resumed.load(std::memory_order_relaxed) == waiterCount,
                "shared counter lost or duplicated coroutine wakeups");

        JobCounter throwingCounter;
        JobTask throwingJob = JobTask::make([]
        {
            throw std::runtime_error("expected raw job failure");
        });
        jobs.kickTasks(&throwingJob, 1, &throwingCounter);
        jobs.waitForCounter(&throwingCounter);
        require(throwingCounter.done(),
                "throwing raw job did not retire its counter");
    }

    void stressMoveOnlyTaskStorage(JobSystem& jobs)
    {
        static_assert(!std::is_copy_constructible_v<JobTask>);
        static_assert(!std::is_copy_assignable_v<JobTask>);

        std::atomic<uint32_t> observed{ 0 };
        auto payload = std::make_unique<uint32_t>(41);
        std::array<std::byte, 256> oversized{};
        JobTask task = JobTask::make(
            [value = std::move(payload), oversized, &observed]() mutable
            {
                observed.store(*value + static_cast<uint32_t>(oversized.size()),
                    std::memory_order_release);
            });
        JobCounter counter;
        jobs.kickTasks(&task, 1, &counter);
        jobs.waitForCounter(&counter);
        require(observed.load(std::memory_order_acquire) == 297,
            "move-only oversized task storage corrupted its capture");
    }

    void stressLocalOverflowAndStealing(JobSystem& jobs)
    {
        constexpr uint32_t childCount = 8192; // deliberately exceeds local capacity
        std::atomic<uint32_t> completed{ 0 };
        std::atomic<uint32_t> active{ 0 };
        std::atomic<uint32_t> maxActive{ 0 };

        JobTask root = JobTask::make([&]
        {
            std::vector<JobTask> children;
            children.reserve(childCount);
            for (uint32_t i = 0; i < childCount; ++i)
            {
                children.push_back(JobTask::make([&]
                {
                    const uint32_t now = active.fetch_add(
                        1, std::memory_order_acq_rel) + 1;
                    uint32_t maximum = maxActive.load(std::memory_order_relaxed);
                    while (maximum < now && !maxActive.compare_exchange_weak(
                        maximum, now, std::memory_order_relaxed)) {}
                    for (uint32_t spin = 0; spin < 64; ++spin)
                    {
                        std::atomic_signal_fence(std::memory_order_seq_cst);
                    }
                    active.fetch_sub(1, std::memory_order_acq_rel);
                    completed.fetch_add(1, std::memory_order_relaxed);
                }));
            }
            JobCounter childrenDone;
            jobs.kickTasks(children.data(), childCount, &childrenDone);
            jobs.waitForCounter(&childrenDone); // nested worker-side helping
        });

        JobCounter rootDone;
        jobs.kickTasks(&root, 1, &rootDone);
        rootDone.wait(); // ensure a real worker owns the local child deque
        require(completed.load(std::memory_order_relaxed) == childCount,
            "local deque overflow lost or duplicated work");
        require(maxActive.load(std::memory_order_relaxed) > 1,
            "local work was not stolen by another worker");
    }

    void stressExternalProducerPressure(JobSystem& jobs)
    {
        constexpr uint32_t producerCount = 8;
        constexpr uint32_t jobsPerProducer = 4096;
        std::atomic<uint32_t> completed{ 0 };
        std::array<std::thread, producerCount> producers;

        for (std::thread& producer : producers)
        {
            producer = std::thread([&]
            {
                std::vector<JobTask> tasks;
                tasks.reserve(jobsPerProducer);
                for (uint32_t i = 0; i < jobsPerProducer; ++i)
                {
                    tasks.push_back(JobTask::make([&completed]
                    {
                        completed.fetch_add(1, std::memory_order_relaxed);
                    }));
                }
                JobCounter counter;
                jobs.kickTasks(tasks.data(), jobsPerProducer, &counter);
                counter.wait();
            });
        }
        for (std::thread& producer : producers) producer.join();
        require(completed.load(std::memory_order_relaxed) ==
                producerCount * jobsPerProducer,
            "concurrent external producers lost or duplicated work");
    }

    void stressSleepWakeRaces(JobSystem& jobs)
    {
        constexpr uint32_t rounds = 2000;
        std::atomic<uint32_t> completed{ 0 };
        for (uint32_t round = 0; round < rounds; ++round)
        {
            if ((round & 31u) == 0u)
            {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
            JobTask task = JobTask::make([&completed]
            {
                completed.fetch_add(1, std::memory_order_relaxed);
            });
            JobCounter counter;
            jobs.kickTasks(&task, 1, &counter);
            counter.wait();
        }
        require(completed.load(std::memory_order_relaxed) == rounds,
            "sleep/wakeup race lost work");
    }

    void stressShutdownDuringCoroutineChain()
    {
        JobSystem jobs;
        jobs.init(4);
        std::atomic<bool> started{ false };
        std::atomic<uint32_t> completed{ 0 };
        std::exception_ptr waiterError;

        std::thread waiter([&]
        {
            try
            {
                sync_wait(jobs, drainingCoroutineChain(jobs, started, completed));
            }
            catch (...)
            {
                waiterError = std::current_exception();
            }
        });

        started.wait(false, std::memory_order_acquire);
        jobs.shutdown();
        waiter.join();
        require(!waiterError,
            "accepted coroutine chain failed while the scheduler drained");
        require(completed.load(std::memory_order_relaxed) == 128u * 32u,
            "shutdown abandoned an accepted coroutine continuation");
    }

    void stressDrainRejectsExternalWork()
    {
        JobSystem jobs;
        jobs.init(2);
        std::atomic<bool> entered{ false };
        std::atomic<bool> release{ false };
        JobTask blocker = JobTask::make([&]
        {
            entered.store(true, std::memory_order_release);
            entered.notify_all();
            release.wait(false, std::memory_order_acquire);
        });
        jobs.kickTasks(&blocker, 1);
        entered.wait(false, std::memory_order_acquire);

        std::thread stopper([&] { jobs.shutdown(); });
        while (jobs.lifecycle() == JobSystem::Lifecycle::Running)
        {
            std::this_thread::yield();
        }

        bool rejected = false;
        try
        {
            JobTask late = JobTask::make([] {});
            jobs.kickTasks(&late, 1);
        }
        catch (const std::runtime_error&)
        {
            rejected = true;
        }
        release.store(true, std::memory_order_release);
        release.notify_all();
        stopper.join();
        require(rejected, "draining scheduler accepted new external work");
        require(jobs.lifecycle() == JobSystem::Lifecycle::Stopped,
            "scheduler did not reach Stopped after draining");
    }

    void stressRepeatedReinitialization()
    {
        JobSystem jobs;
        std::atomic<uint32_t> completed{ 0 };
        for (uint32_t round = 0; round < 32; ++round)
        {
            jobs.init(3);
            std::array<JobTask, 64> tasks;
            for (JobTask& task : tasks)
            {
                task = JobTask::make([&completed]
                {
                    completed.fetch_add(1, std::memory_order_relaxed);
                });
            }
            JobCounter counter;
            jobs.kickTasks(tasks.data(), static_cast<uint32_t>(tasks.size()), &counter);
            jobs.shutdown();
            require(counter.done(), "reinitialized scheduler abandoned its batch");
        }
        require(completed.load(std::memory_order_relaxed) == 32u * 64u,
            "reinitialization lost accepted work");
    }

#if defined(NDEBUG)
    // Retained A/B reference for the architecture this refactor replaces: one
    // BlockingConcurrentQueue shared by every producer and worker. It uses the
    // same move-only JobTask payload so the comparison isolates scheduling.
    class GlobalQueueBaseline
    {
    public:
        explicit GlobalQueueBaseline(uint32_t workerCount)
        {
            for (uint32_t i = 0; i < workerCount; ++i)
            {
                m_workers.emplace_back([this]
                {
                    JobTask task;
                    while (m_running.load(std::memory_order_acquire))
                    {
                        if (m_queue.try_dequeue(task))
                        {
                            if (task.function) task.function();
                            task = JobTask{};
                        }
                        else
                        {
                            std::this_thread::yield();
                        }
                    }
                });
            }
        }

        ~GlobalQueueBaseline()
        {
            m_running.store(false, std::memory_order_release);
            for (std::thread& worker : m_workers) worker.join();
        }

        void submit(JobTask&& task)
        {
            require(m_queue.enqueue(std::move(task)),
                "global baseline enqueue failed");
        }

        void helpUntil(std::atomic<uint32_t>& value, uint32_t target)
        {
            JobTask task;
            while (value.load(std::memory_order_acquire) != target)
            {
                if (m_queue.try_dequeue(task))
                {
                    if (task.function) task.function();
                    task = JobTask{};
                }
                else
                {
                    std::this_thread::yield();
                }
            }
        }

    private:
        moodycamel::BlockingConcurrentQueue<JobTask> m_queue { 1024 };
        std::vector<std::thread> m_workers;
        std::atomic<bool> m_running { true };
    };

    using BenchmarkClock = std::chrono::steady_clock;

    void waitForRoots(std::atomic<uint32_t>& started, uint32_t rootCount)
    {
        uint32_t value = started.load(std::memory_order_acquire);
        while (value != rootCount)
        {
            started.wait(value, std::memory_order_acquire);
            value = started.load(std::memory_order_acquire);
        }
    }

    double benchmarkHybridBatch(
        JobSystem& jobs, uint32_t rootCount, uint32_t childrenPerRoot)
    {
        std::atomic<uint32_t> started{ 0 };
        std::atomic<uint32_t> completed{ 0 };
        std::vector<JobTask> roots;
        roots.reserve(rootCount);
        for (uint32_t rootIndex = 0; rootIndex < rootCount; ++rootIndex)
        {
            roots.push_back(JobTask::make([&]
            {
                if (started.fetch_add(1, std::memory_order_acq_rel) + 1 == rootCount)
                {
                    started.notify_all();
                }
                waitForRoots(started, rootCount);

                std::vector<JobTask> children;
                children.reserve(childrenPerRoot);
                for (uint32_t i = 0; i < childrenPerRoot; ++i)
                {
                    children.push_back(JobTask::make([&completed]
                    {
                        completed.fetch_add(1, std::memory_order_relaxed);
                    }));
                }
                JobCounter counter;
                jobs.kickTasks(children.data(), childrenPerRoot, &counter);
                jobs.waitForCounter(&counter);
            }));
        }

        JobCounter rootsDone;
        const auto start = BenchmarkClock::now();
        jobs.kickTasks(roots.data(), rootCount, &rootsDone);
        waitForRoots(started, rootCount);
        jobs.waitForCounter(&rootsDone);
        const auto end = BenchmarkClock::now();
        require(completed.load(std::memory_order_relaxed) ==
                rootCount * childrenPerRoot,
            "hybrid benchmark lost work");
        return std::chrono::duration<double, std::micro>(end - start).count();
    }

    double benchmarkGlobalBatch(
        GlobalQueueBaseline& jobs, uint32_t rootCount, uint32_t childrenPerRoot)
    {
        std::atomic<uint32_t> started{ 0 };
        std::atomic<uint32_t> completed{ 0 };
        std::atomic<uint32_t> rootsDone{ 0 };
        std::vector<JobTask> roots;
        roots.reserve(rootCount);
        for (uint32_t rootIndex = 0; rootIndex < rootCount; ++rootIndex)
        {
            roots.push_back(JobTask::make([&]
            {
                if (started.fetch_add(1, std::memory_order_acq_rel) + 1 == rootCount)
                {
                    started.notify_all();
                }
                waitForRoots(started, rootCount);

                std::atomic<uint32_t> localCompleted{ 0 };
                for (uint32_t i = 0; i < childrenPerRoot; ++i)
                {
                    jobs.submit(JobTask::make([&localCompleted, &completed]
                    {
                        completed.fetch_add(1, std::memory_order_relaxed);
                        localCompleted.fetch_add(1, std::memory_order_release);
                    }));
                }
                jobs.helpUntil(localCompleted, childrenPerRoot);
                rootsDone.fetch_add(1, std::memory_order_release);
            }));
        }

        const auto start = BenchmarkClock::now();
        for (JobTask& root : roots) jobs.submit(std::move(root));
        waitForRoots(started, rootCount);
        jobs.helpUntil(rootsDone, rootCount);
        const auto end = BenchmarkClock::now();
        require(completed.load(std::memory_order_relaxed) ==
                rootCount * childrenPerRoot,
            "global benchmark lost work");
        return std::chrono::duration<double, std::micro>(end - start).count();
    }

    void benchmarkAgainstGlobalQueue()
    {
        constexpr uint32_t childrenPerRoot = 1024;
        constexpr uint32_t sampleCount = 11;
        const uint32_t workerCount = std::max(
            2u, std::thread::hardware_concurrency());
        JobSystem hybrid;
        hybrid.init(workerCount);
        GlobalQueueBaseline global(workerCount);

        // Warm caches and worker parking state before collecting samples.
        (void)benchmarkHybridBatch(hybrid, workerCount, 64);
        (void)benchmarkGlobalBatch(global, workerCount, 64);

        std::array<double, sampleCount> hybridUs{};
        std::array<double, sampleCount> globalUs{};
        for (uint32_t i = 0; i < sampleCount; ++i)
        {
            // Alternate order to avoid consistently favoring the first run.
            if ((i & 1u) == 0u)
            {
                hybridUs[i] = benchmarkHybridBatch(
                    hybrid, workerCount, childrenPerRoot);
                globalUs[i] = benchmarkGlobalBatch(
                    global, workerCount, childrenPerRoot);
            }
            else
            {
                globalUs[i] = benchmarkGlobalBatch(
                    global, workerCount, childrenPerRoot);
                hybridUs[i] = benchmarkHybridBatch(
                    hybrid, workerCount, childrenPerRoot);
            }
        }
        hybrid.shutdown();

        std::sort(hybridUs.begin(), hybridUs.end());
        std::sort(globalUs.begin(), globalUs.end());
        const double hybridMedian = hybridUs[sampleCount / 2];
        const double globalMedian = globalUs[sampleCount / 2];
        const double hybridP95 = hybridUs[sampleCount - 1];
        const double globalP95 = globalUs[sampleCount - 1];
        const double speedup = globalMedian / hybridMedian;
        const uint32_t totalJobs = workerCount * childrenPerRoot;
        std::cout << "Scheduler A/B concurrent nested tiny-job fan-out ("
                  << totalJobs
                  << " jobs, " << workerCount << " workers): hybrid median="
                  << hybridMedian << " us p95=" << hybridP95
                  << " us, global median=" << globalMedian << " us p95="
                  << globalP95 << " us, speedup=" << speedup << "x\n";

        // A 10% median gain is large enough to clear ordinary run-to-run noise
        // and justify retaining the more sophisticated scheduler.
        require(speedup >= 1.10,
            "hybrid scheduler did not materially outperform global queue");
    }
#endif

    void stressCallerThreadHelping()
    {
        // An uninitialized system has no workers. sync_wait must still drain the
        // queue itself without deadlocking.
        JobSystem jobs;
        std::atomic<uint32_t> completed{ 0 };
        std::vector<JobTask> tasks;

        tasks.reserve(4096);

        for (uint32_t i = 0; i < 4096; ++i)
        {
            tasks.push_back(JobTask::make([&]
                {
                    completed.fetch_add(1, std::memory_order_relaxed);
                }));
        }
        JobCounter counter;
        sync_wait(jobs, awaitJobs(jobs, tasks, counter));
        require(completed.load(std::memory_order_relaxed) == tasks.size(),
                "caller-thread queue helping deadlocked or lost work");
    }

    void stressLifecycle()
    {
        for (uint32_t iteration = 0; iteration < 50; ++iteration)
        {
            JobSystem jobs;
            jobs.init(2);
            std::atomic<uint32_t> completed{ 0 };
            std::array<JobTask, 128> tasks;
            for (JobTask& task : tasks)
            {
                task = JobTask::make([&]
                {
                    completed.fetch_add(1, std::memory_order_relaxed);
                });
            }
            JobCounter counter;
            jobs.kickTasks(tasks.data(), static_cast<uint32_t>(tasks.size()), &counter);
            jobs.shutdown();
            jobs.shutdown();
            require(completed.load(std::memory_order_relaxed) == tasks.size(),
                    "job-system lifecycle lost work");
            require(counter.done(), "shutdown abandoned queued counter work");
        }
    }
}

namespace
{
    // Per-test progress. Printed to stderr and flushed so that if a run wedges,
    // the last "[begin]" line without a matching "[ ok  ]" names the culprit.
    // This makes any timeout in the repeated-run harness attributable.
    template<typename Fn>
    void runPhase(const char* name, Fn&& fn)
    {
        const auto phaseStart = std::chrono::steady_clock::now();
        std::cerr << "[begin] " << name << std::endl;
        std::forward<Fn>(fn)();
        const auto phaseMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - phaseStart).count();
        std::cerr << "[ ok  ] " << name << " (" << phaseMs << " ms)" << std::endl;
    }
}

int runJobSystemStressTests()
{
    const auto start = std::chrono::steady_clock::now();

    runPhase("callerThreadHelping",        [] { stressCallerThreadHelping(); });
    runPhase("lifecycle",                  [] { stressLifecycle(); });
    runPhase("repeatedReinitialization",   [] { stressRepeatedReinitialization(); });
    runPhase("drainRejectsExternalWork",   [] { stressDrainRejectsExternalWork(); });
    runPhase("shutdownDuringCoroutineChain",[] { stressShutdownDuringCoroutineChain(); });

    ic::JobSystem jobs;
    jobs.init(std::max(2u, std::thread::hardware_concurrency()));
    runPhase("rawJobs",                    [&] { stressRawJobs(jobs); });
    runPhase("counterReuse",               [&] { stressCounterReuse(jobs); });
    runPhase("coroutines",                 [&] { stressCoroutines(jobs); });
    runPhase("concurrentCounterEpisodes",  [&] { stressConcurrentCounterEpisodes(jobs); });
    runPhase("sharedWaitersAndJobExceptions",[&] { stressSharedWaitersAndJobExceptions(jobs); });
    runPhase("moveOnlyTaskStorage",        [&] { stressMoveOnlyTaskStorage(jobs); });
    runPhase("localOverflowAndStealing",   [&] { stressLocalOverflowAndStealing(jobs); });
    runPhase("externalProducerPressure",   [&] { stressExternalProducerPressure(jobs); });
    runPhase("sleepWakeRaces",             [&] { stressSleepWakeRaces(jobs); });
    jobs.shutdown();

#if defined(NDEBUG)
    runPhase("benchmarkAgainstGlobalQueue",[] { benchmarkAgainstGlobalQueue(); });
#endif

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    std::cout << "Job/coroutine stress tests passed in "
              << elapsed.count() << " ms\n";
    return 0;
}
