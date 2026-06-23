#include "ic/common/ic_pch.h"
#include "ic/core/job_system.h"
#include <spdlog/spdlog.h>

namespace ic 
{

    void JobSystem::init(uint32_t threadCount)
    {
        if (threadCount == 0)
        {
            // Leave one core free for OS/driver maintenance overhead
            threadCount = std::max(1u, std::thread::hardware_concurrency() - 1);
        }

        m_running = true;
        m_workers.reserve(threadCount);

        for (uint32_t i = 0; i < threadCount; ++i)
        {
            m_workers.emplace_back(&JobSystem::workerLoop, this);
        }
    }

    void JobSystem::shutdown()
    {
        m_running = false;

        for (auto& thread : m_workers)
        {
            if (thread.joinable()) thread.join();
        }
        m_workers.clear();
    }

    void JobSystem::kickTasks(const JobTask* tasks, uint32_t count, JobCounter* counter)
    {
        if (counter)
        {
            counter->increment(count);
        }

        for (uint32_t i = 0; i < count; ++i)
        {
            m_taskQueue.enqueue(tasks[i]);
        }
    }

    void JobSystem::waitForCounter(JobCounter* counter)
    {
        if (!counter || counter->value() == 0) return;

        JobTask task;
        while (counter->value() > 0)
        {
            if (m_taskQueue.try_dequeue(task))
            {
                task.function(task.payload);
                counter->decrement();
            }
            else
            {
                std::this_thread::yield();
            }
        }
    }

    void JobSystem::workerLoop()
    {
        JobTask task;
        while (m_running)
        {
            if (m_taskQueue.try_dequeue(task))
            {
                task.function(task.payload);
            }
            else
            {
                std::this_thread::yield();
            }
        }
    }
}