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

        spdlog::info("[JobSystem] Starting {} worker threads", threadCount);

        static constexpr size_t kQueuePrealloc = 1024;
        m_taskQueue = moodycamel::ConcurrentQueue<JobTask>(kQueuePrealloc);


        m_running = true;
        m_workers.reserve(threadCount);

        for (uint32_t i = 0; i < threadCount; ++i)
        {
            m_workers.emplace_back(&JobSystem::workerLoop, this);

#if defined(_WIN32)
            auto handle = m_workers.back().native_handle();
            std::wstring name = L"JobWorker-" + std::to_wstring(i);
            SetThreadDescription(handle, name.c_str());
            spdlog::info("[JobSystem]   JobWorker-{} started", i);
#elif defined(__linux__)
            auto handle = m_workers.back().native_handle();
            std::string name = "JobWorker-" + std::to_string(i);
            pthread_setname_np(handle, name.c_str());
            spdlog::info("[JobSystem]   JobWorker-{} started", i);
#else
            spdlog::info("[JobSystem]   Worker-{} started (unnamed platform)", i);
#endif
        }
    }

    void JobSystem::shutdown()
    {
        m_running = false;

        // Wake all workers so they can observe m_running == false
        // And exit cleanly

        m_semaphore.release(static_cast<std::ptrdiff_t>(m_workers.size()));

        for (auto& thread : m_workers)
        {
            if (thread.joinable()) thread.join();
        }
        m_workers.clear();
        spdlog::info("[JobSystem] All workers joined");
    }

    void JobSystem::kickTasks(const JobTask* tasks, uint32_t count, JobCounter* counter)
    {
        if (count == 0) return;

        if (counter)
        {
            counter->increment(count);
        }

        for (uint32_t i = 0; i < count; ++i)
        {
            JobTask t = tasks[i];
            t.counter = counter;
            m_taskQueue.enqueue(t);
        }

        // Wake up sleep workers, one relase per task.
        // Workers that are already running don't block on the semaphore,
        // so this won't over wake
        m_semaphore.release(static_cast<ptrdiff_t>(count));
    }

    void JobSystem::waitForCounter(JobCounter* counter)
    {
        if (!counter || counter->done()) return;

        JobTask task;
        uint32_t spinCount = 0;
        while (!counter->done())
        {
            if (m_taskQueue.try_dequeue(task))
            {
                [[maybe_unused]] bool slotConsumed =
                    m_semaphore.try_acquire();

                executeTask(task);
            }
            else
            {
                if (++spinCount < 16) 
                    _mm_pause(); // I should wrap this in a platform macro for ARM portability
                else 
                    std::this_thread::yield();
            }
        }
    }

    void JobSystem::executeTask(JobTask& task)
    {
        task.invoke(task.storage);
        if (task.destroy) task.destroy(task.storage);
        if (task.counter) task.counter->decrement();
    }

    void JobSystem::workerLoop()
    {
        JobTask task;
        while (true)
        {
            m_semaphore.acquire();
            if (!m_running) break;
            if (m_taskQueue.try_dequeue(task))
            {
                executeTask(task);
            }
        }
    }
}