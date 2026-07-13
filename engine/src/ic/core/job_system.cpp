#include "ic/common/ic_pch.h"
#include "ic/core/job_system.h"
#include "ic/util/profiler.h"
#include <spdlog/spdlog.h>

namespace ic 
{

    void JobSystem::init(uint32_t threadCount)
    {
        if (m_running.exchange(true, std::memory_order_acq_rel))
        {
            return;
        }

        if (threadCount == 0)
        {
            // Leave one core free for OS/driver maintenance overhead
            const uint32_t hardwareThreads = std::thread::hardware_concurrency();
            threadCount = hardwareThreads > 1 ? hardwareThreads - 1 : 1;
        }

        spdlog::info("[JobSystem] Starting {} worker threads", threadCount);

        static constexpr size_t kQueuePrealloc = 1024;
        m_taskQueue = moodycamel::ConcurrentQueue<JobTask>(kQueuePrealloc);


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
        if (!m_running.exchange(false, std::memory_order_acq_rel))
        {
            return;
        }

        // Wake all workers so they can observe m_running == false
        // And exit cleanly

        m_semaphore.release(static_cast<std::ptrdiff_t>(m_workers.size()));

        for (auto& thread : m_workers)
        {
            if (thread.joinable()) thread.join();
        }
        m_workers.clear();
        m_workEpoch.fetch_add(1, std::memory_order_release);
        m_workEpoch.notify_all();
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
        m_workEpoch.fetch_add(1, std::memory_order_release);
        m_workEpoch.notify_all();
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

    void JobSystem::scheduleResume(std::coroutine_handle<> handle)
    {
        ZoneScopedN("JobSystem::scheduleResume");
        if (!handle) return;

        // A coroutine resume rides on the normal job queue as an ordinary task
        // with no counter. Capturing the handle by value keeps the payload
        // trivially copyable into the queue; the frame it refers to is owned
        // elsewhere (by a Task<>, or self-owned until final_suspend destroys it).
        JobTask t;
        t.function = [handle]() { handle.resume(); };
        t.counter  = nullptr;
        t.publishCompletion = true;

        m_taskQueue.enqueue(t);
        m_semaphore.release(1);
        m_workEpoch.fetch_add(1, std::memory_order_release);
        m_workEpoch.notify_all();
    }

    bool JobSystem::tryRunOne()
    {
        JobTask task;
        if (m_taskQueue.try_dequeue(task))
        {
            // Keep the semaphore count consistent with the queue depth: we just
            // consumed a task a sleeping worker was going to be woken for.
            [[maybe_unused]] bool slotConsumed = m_semaphore.try_acquire();
            executeTask(task);
            return true;
        }
        return false;
    }

    void JobCounter::notifyWaiters()
    {
        // Close the list: atomically swap the head to the sentinel and take the
        // parked nodes. Any awaiter that races a push after this point sees the
        // sentinel and resumes inline instead of parking (no lost wakeup); any
        // awaiter mid-push either lands before us (we schedule it) or its CAS
        // fails against the sentinel and it resumes inline. Exactly-once wake.
        //
        // The exchange also publishes (release) everything the finished work
        // wrote; the worker that dequeues the resume acquires it before running
        // the coroutine — so the resumed coroutine sees a consistent view with
        // no data race.
        CounterWaitNode* node = m_waiters.exchange(closedSentinel(),
                                                   std::memory_order_acq_rel);
        if (node == closedSentinel())
        {
            return; // already closed by a prior zero-transition
        }

        while (node)
        {
            CounterWaitNode* next = node->next;   // read next before scheduling:
                                                  // once resumed, the node (which
                                                  // lives in the coroutine frame)
                                                  // may be destroyed underneath us.
            if (node->system)
            {
                node->system->scheduleResume(node->handle);
            }
            node = next;
        }
    }

    void JobSystem::executeTask(JobTask& task)
    {
        try
        {
            if (task.function)
            {
                task.function();
            }
        }
        catch (const std::exception& error)
        {
            // Jobs cannot propagate across a worker-thread boundary. Keep the
            // pool alive and, critically, still retire the associated counter.
            spdlog::error("[JobSystem] Unhandled job exception: {}", error.what());
        }
        catch (...)
        {
            spdlog::error("[JobSystem] Unhandled non-standard job exception");
        }
        if (task.counter) task.counter->decrement();

        if (task.publishCompletion)
        {
            // The final continuation may already have been dequeued before a
            // sync_wait caller samples the epoch. Publish its completion so the
            // caller cannot sleep indefinitely. Regular jobs deliberately skip
            // this cache-line write and wake storm.
            m_workEpoch.fetch_add(1, std::memory_order_release);
            m_workEpoch.notify_all();
        }
    }

    void JobSystem::workerLoop()
    {
        JobTask task;
        while (true)
        {
            m_semaphore.acquire();
            if (m_taskQueue.try_dequeue(task))
            {
                executeTask(task);
                continue;
            }

            // shutdown() contributes one token per worker after closing the
            // system. Workers consume all real queued tokens first, including
            // continuations spawned while draining, and only exit when a
            // shutdown token finds the queue empty.
            if (!m_running.load(std::memory_order_acquire))
            {
                break;
            }
        }
    }
}
