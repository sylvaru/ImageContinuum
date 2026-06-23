#pragma once
#include <vector>
#include <thread>
#include <atomic>
#include <concurrentqueue.h>

namespace ic 
{

    /*

    This Job System is responsible for doing expensive things like
    executing frame DAG nodes

    */

    using JobTaskFunction = void(*)(void* payload);

    struct JobTask
    {
        JobTaskFunction function{ nullptr };
        void* payload{ nullptr };
    };

    class JobCounter
    {
    public:
        JobCounter() : m_value(0) {}

        void increment(uint32_t count = 1) 
        { m_value.fetch_add(count, std::memory_order_release); }

        void decrement(uint32_t count = 1) 
        { m_value.fetch_sub(count, std::memory_order_release); }

        uint32_t value() const 
        { return m_value.load(std::memory_order_acquire); }

    private:
        std::atomic<uint32_t> m_value;
    };

    class JobSystem
    {
    public:
        JobSystem() = default;
        ~JobSystem() = default;

        void init(uint32_t threadCount = 0);
        void shutdown();

        // Push an array of tasks onto the global worker queue
        void kickTasks(const JobTask* tasks, uint32_t count, JobCounter* counter = nullptr);

        // Blocks the calling thread (main thread) 
        // and works on remaining jobs until counter is 0
        void waitForCounter(JobCounter* counter);

    private:
        void workerLoop();

        std::vector<std::thread> m_workers;
        moodycamel::ConcurrentQueue<JobTask> m_taskQueue;
        std::atomic<bool> m_running{ false };
    };
}

