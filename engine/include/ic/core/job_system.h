#pragma once
#include <vector>
#include <thread>
#include <atomic>
#include <semaphore>
#include <functional>
#include <coroutine>
#include <cstdint>
#include <climits>
#include <concurrentqueue.h>

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

        Design notes:
          Workers sleep via std::counting_semaphore; no spin yield on idle.
          Counter decrement happens in executeTask(), shared by both the
          worker path and the main thread work stealing path in waitForCounter.
          kickTasks releases the semaphore exactly once per task so workers
          wake proportionally to work arriving.

    */

    class JobCounter
    {
    public:
        JobCounter() : m_value(0) {}

        void increment(uint32_t count = 1)
        {
            const uint32_t prev = m_value.fetch_add(count, std::memory_order_release);

            // Re arm the coroutine waiter list at the start of a fresh wait
            // episode (0  > positive). A prior episode may have left m_waiters at
            // the "closed" sentinel; clearing it here lets new awaiters park
            // again. This mirrors the counter's documented lifecycle: kick
            // (increment) fully, then wait — the same contract waitForCounter
            // already relies on.
            if (prev == 0)
            {
                m_waiters.store(nullptr, std::memory_order_release);
            }
        }

        void decrement(uint32_t count = 1)
        {
            const uint32_t prev = m_value.fetch_sub(count, std::memory_order_acq_rel);

            // Zero-transition: the decrement that consumes the last outstanding
            // unit must LATCH the waiter list closed (and drain any parked
            // coroutines).
            //
            // Critically, this has to run even when no waiter is parked *yet*.
            // await_ready() and await_suspend() in CounterAwaiter are not atomic
            // together: an awaiter can observe value > 0 in await_ready(), then a
            // worker drives the counter to zero here, and only afterwards does
            // await_suspend() push its node. If this transition did not latch the
            // list closed, that late node would park on a counter that will never
            // decrement again — a lost wakeup that freezes the coroutine (and,
            // via recordFrameGraph, the whole frame). Latching closed makes the
            // late pushWaiter() observe the sentinel and resume inline instead.
            //
            // The load-guard only skips the (idempotent) work when the list is
            // already closed; on the first zero-transition m_waiters is null,
            // which is != closed, so notifyWaiters() runs and latches it.
            if (prev == count &&
                m_waiters.load(std::memory_order_acquire) != closedSentinel())
            {
                notifyWaiters();
            }
        }

        uint32_t value() const
        {
            return m_value.load(std::memory_order_acquire);
        }

        bool done() const { return value() == 0; }

        // Coroutine bridge
        //
        // Lock free Treiber stack of parked coroutines with a "closed" sentinel
        // head, following the async manual reset event protocol. The sentinel is
        // what makes suspension race free WITHOUT the awaiter ever having to
        // re touch its own frame after publishing itself — see CounterAwaiter in
        // task.h for why that property is essential (a concurrent wake may have
        // already destroyed the awaiter).
        //
        // Returns true if the node was published (caller stays suspended), false
        // if the counter had already reached zero and closed the list (caller
        // must resume inline; the node was NOT published, so it is safe to keep
        // using the awaiter/frame).
        bool pushWaiter(CounterWaitNode* node)
        {
            CounterWaitNode* head = m_waiters.load(std::memory_order_acquire);
            do
            {
                if (head == closedSentinel())
                {
                    return false; // already zero: do not park, resume inline
                }
                node->next = head;
            }
            while (!m_waiters.compare_exchange_weak(
                head, node,
                std::memory_order_release,
                std::memory_order_acquire));

            return true; // published — MUST NOT touch node/awaiter after this
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

        std::atomic<uint32_t>          m_value;
        std::atomic<CounterWaitNode*>  m_waiters { nullptr };
    };



    struct JobTask
    {
        std::function<void()> function;
        JobCounter* counter { nullptr };
        // Only continuation jobs need to publish completion to sync_wait.
        // Ordinary jobs signal through their JobCounter and avoid this atomic
        // hot-path traffic entirely.
        bool publishCompletion { false };

        template<typename F>
        static JobTask make(F&& fn, JobCounter* ctr = nullptr)
        {
            JobTask t;
            t.counter = ctr;
            t.function = std::forward<F>(fn);
            return t;
        }
    };

    class JobSystem
    {
    public:
        JobSystem() = default;
        ~JobSystem() { shutdown(); }

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

        // Coroutine integration
        //
        // Schedule a coroutine resumption onto the same lock free queue the
        // workers already drain. A resume is just a JobTask whose function calls
        // handle.resume(); the existing worker loop runs it with zero changes.
        // This is the single mechanism by which a suspended coroutine gets put
        // back onto a worker — no thread ever blocks waiting for it.
        void scheduleResume(std::coroutine_handle<> handle);

        // Run at most one queued job on the calling thread if one is available.
        // Returns true if a job was executed. Used by sync_wait() to let a
        // non worker (e.g. the main thread) help drain the queue while it blocks
        // at the coroutine/blocking boundary, mirroring waitForCounter().
        bool tryRunOne();

        // Monotonic notification source used by blocking helpers that also
        // participate in queue draining. Waiting on this avoids a busy-yield
        // loop without reserving a worker semaphore token.
        uint64_t workEpoch() const noexcept
        {
            return m_workEpoch.load(std::memory_order_acquire);
        }
        void waitForWork(uint64_t epoch) const noexcept
        {
            m_workEpoch.wait(epoch, std::memory_order_acquire);
        }

    private:

        // Runs task.function, then decrements task.counter if present.
        // Single authoritative call site for task completion
        void executeTask(JobTask& task);

        void workerLoop();

        std::vector<std::thread>                m_workers;
        moodycamel::ConcurrentQueue<JobTask>    m_taskQueue;

        std::counting_semaphore<INT_MAX>         m_semaphore{ 0 };
        std::atomic<uint64_t>                   m_workEpoch{ 0 };
        std::atomic<bool>                       m_running{ false };
    };
}

