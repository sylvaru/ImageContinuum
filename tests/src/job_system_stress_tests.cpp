#include "common/tests_pch.h"
#include "job_system_stress_tests.h"

#include "ic/core/job_system.h"
#include "ic/core/task.h"

#include <cstdlib>
#include <numeric>
#include <span>

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

    Task<void> awaitJobs(
        JobSystem& jobs,
        std::span<const JobTask> tasks,
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
        std::array<JobTask, tasksPerRound> tasks;
        for (JobTask& task : tasks)
        {
            task = JobTask::make([&]
            {
                completed.fetch_add(1, std::memory_order_relaxed);
            });
        }

        JobCounter counter;
        for (uint32_t round = 0; round < rounds; ++round)
        {
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

        const JobTask release = JobTask::make(
            [&sharedCounter] { sharedCounter.decrement(); });
        jobs.kickTasks(&release, 1);
        sync_wait(jobs, when_all(jobs, std::move(waiters)));
        require(resumed.load(std::memory_order_relaxed) == waiterCount,
                "shared counter lost or duplicated coroutine wakeups");

        JobCounter throwingCounter;
        const JobTask throwingJob = JobTask::make([]
        {
            throw std::runtime_error("expected raw job failure");
        });
        jobs.kickTasks(&throwingJob, 1, &throwingCounter);
        jobs.waitForCounter(&throwingCounter);
        require(throwingCounter.done(),
                "throwing raw job did not retire its counter");
    }

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

int runJobSystemStressTests()
{
    const auto start = std::chrono::steady_clock::now();

    stressCallerThreadHelping();
    stressLifecycle();

    ic::JobSystem jobs;
    jobs.init(std::max(2u, std::thread::hardware_concurrency()));
    stressRawJobs(jobs);
    stressCounterReuse(jobs);
    stressCoroutines(jobs);
    stressSharedWaitersAndJobExceptions(jobs);
    jobs.shutdown();

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    std::cout << "Job/coroutine stress tests passed in "
              << elapsed.count() << " ms\n";
    return 0;
}
