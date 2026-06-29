#pragma once
#include <vector>
#include <thread>
#include <atomic>
#include <semaphore>
#include <concurrentqueue.h>

namespace ic 
{

    /*

        Job System — executes frame DAG nodes and graph passes in parallel.

        Design notes:
        - Workers sleep via std::counting_semaphore; no spin-yield on idle.
        - Counter decrement happens in executeTask(), shared by both the
          worker path and the main-thread work-stealing path in waitForCounter.
        - kickTasks releases the semaphore exactly once per task so workers
          wake proportionally to work arriving.

    */

    class JobCounter
    {
    public:
        JobCounter() : m_value(0) {}

        void increment(uint32_t count = 1)
        {
            m_value.fetch_add(count, std::memory_order_release);
        }

        void decrement(uint32_t count = 1)
        {
            m_value.fetch_sub(count, std::memory_order_acq_rel);
        }

        uint32_t value() const
        {
            return m_value.load(std::memory_order_acquire);
        }

        bool done() const { return value() == 0; }

    private:
        std::atomic<uint32_t> m_value;
    };



    static constexpr size_t kJobInlineStorageBytes = 64;

    struct JobTask
    {
        alignas(8) std::byte storage[kJobInlineStorageBytes];
        void (*invoke)(std::byte*)  { nullptr };
        void (*destroy)(std::byte*) { nullptr };
        JobCounter* counter         { nullptr };

        template<typename F>
        static JobTask make(F&& fn, JobCounter* ctr = nullptr)
        {
            static_assert(sizeof(F) <= kJobInlineStorageBytes,
                "JobTask closure too large — increase kJobInlineStorageBytes "
                "or reduce captures");
            static_assert(std::is_trivially_destructible_v<F> || true,
                "Non-trivial destructors supported via destroy ptr");

            JobTask t;
            t.counter = ctr;
            new (t.storage) F(std::forward<F>(fn));
            t.invoke = [](std::byte* s) {(*std::launder(reinterpret_cast<F*>(s)))(); };
            t.destroy = [](std::byte* s) { std::launder(reinterpret_cast<F*>(s))->~F(); };
            return t;
        }
    };

    class JobSystem
    {
    public:
        JobSystem() = default;
        ~JobSystem() = default;

        void init(uint32_t threadCount = 0);
        void shutdown();

        // Push an array of tasks onto the global worker queue.
        // Counter is incremented here 
        // and decremented automatically when each task completes
                 
        void kickTasks(
            const JobTask* tasks,
            uint32_t count,
            JobCounter* counter = nullptr);

        // Blocks the calling thread (main thread) 
        // and works on remaining jobs until counter is 0
        void waitForCounter(JobCounter* counter);

        uint32_t workerCount() const
        {
            return static_cast<uint32_t>(m_workers.size());
        }

    private:

        // Runs task.function, then decrements task.counter if present.
        // Single authoritative call site for task completion
        void executeTask(JobTask& task);

        void workerLoop();

        std::vector<std::thread>                m_workers;
        moodycamel::ConcurrentQueue<JobTask>    m_taskQueue;

        std::counting_semaphore<(1 << 20)>      m_semaphore{ 0 }; // semaphore capped at max realistic tasks in flight
        std::atomic<bool>                       m_running{ false };
    };
}

