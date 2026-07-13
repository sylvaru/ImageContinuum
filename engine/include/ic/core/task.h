// ic/core/task.h
#pragma once

#include "ic/core/job_system.h"
#include "ic/core/coroutine_frame_pool.h"
#include "ic/util/profiler.h"

#include <coroutine>
#include <exception>
#include <utility>
#include <type_traits>
#include <memory>
#include <new>
#include <cstddef>
#include <tuple>
#include <vector>
#include <atomic>
#include <optional>
#include <thread>
#include <variant>
#include <concepts>

namespace ic
{
    template<typename T>
    concept NonVoid = !std::same_as<std::remove_cv_t<T>, void>;

    /*

        Task<T> — a C++20 coroutine type layered on top of the existing
        JobSystem (moodycamel::ConcurrentQueue + counting_semaphore workers).

        Coroutines are strictly ADDITIVE: JobTask / kickTasks / waitForCounter
        are untouched. A coroutine resume is simply a JobTask whose body calls
        handle.resume(), so the existing worker loop runs it verbatim.

        Three properties the design guarantees, and *why*:

        1. No busy-waiting, suspension frees the worker.
           When a coroutine awaits something not yet ready it returns control to
           the worker loop (await_suspend returns noop_coroutine or bool=true),
           so the worker immediately picks up other jobs. Nothing spins; nothing
           blocks a worker thread on behalf of a suspended coroutine.

        2. Symmetric transfer — no stack growth on chained continuations.
           `co_await childTask` STARTS the child by having the awaiter's
           await_suspend RETURN the child's coroutine_handle (symmetric
           transfer). The compiler tail-resumes the child instead of nesting a
           resume() call, so a chain A -> B -> C -> ... does not grow the stack.
           On completion, the child's final_suspend hands its parent back to the
           job queue (scheduleResume) and returns noop_coroutine(): the worker
           unwinds to its loop (again, no growth) and the parent is resumed by
           whichever worker dequeues it.

        3. No use-after-free on cross-thread resume.
           A completed child SUSPENDS at final_suspend (it does not self-destroy),
           so its frame — and therefore its result — stays alive until the owning
           Task<T> is destroyed. The parent reads that result in await_resume on
           the resuming worker; the child frame is only destroyed later when the
           child Task<T> temporary goes out of scope (end of the co_await
           full-expression), strictly after the read. The queue enqueue/dequeue
           pair provides the release/acquire edge that makes the finished work's
           writes visible to the resuming worker.

    */

    template<typename T>
    class Task;

    namespace detail
    {
        template<typename Promise, typename T>
        class TaskReturnObject
        {
        public:
            Task<T> get_return_object() noexcept;
        };

        // Awaiter returned from every task promise's final_suspend().
        //
        // Resumes the continuation by SCHEDULING it back onto the job system
        // (requirement: "schedules continuations back onto the job system rather
        // than blocking a worker"), returning noop_coroutine() so the finishing
        // worker unwinds to its loop without stack growth. If no system is set
        // (a task driven purely by inline symmetric transfer, e.g. under
        // sync_wait with no pool), it falls back to symmetric transferring into
        // the continuation directly.
        struct FinalAwaiter
        {
            bool await_ready() const noexcept { return false; }

            template<typename Promise>
            std::coroutine_handle<> await_suspend(std::coroutine_handle<Promise> finished) noexcept
            {
                ZoneScopedN("Task::final_suspend");

                auto&                   promise      = finished.promise();
                std::coroutine_handle<> continuation = promise.continuation();

                if (!continuation)
                {
                    // Top level task with nobody waiting: park here. The owning
                    // Task<T> (or sync_wait) observes done() and destroys us.
                    return std::noop_coroutine();
                }

                if (JobSystem* system = promise.system())
                {
                    // Cross-thread-safe hand-off: publish the continuation to the
                    // queue and let a worker pick it up. We must not touch this
                    // frame afterward — treat it as owned by the resuming worker.
                    system->scheduleResume(continuation);
                    return std::noop_coroutine();
                }

                // No job system in play: resume inline via symmetric transfer.
                return continuation;
            }

            void await_resume() const noexcept {}
        };

        // Common promise state shared by the value and void specializations.
        class TaskPromiseBase
        {
        public:
            std::suspend_always initial_suspend() noexcept { return {}; }
            FinalAwaiter        final_suspend() noexcept { return {}; }

            void setContinuation(std::coroutine_handle<> c) noexcept { m_continuation = c; }
            std::coroutine_handle<> continuation() const noexcept { return m_continuation; }

            void       setSystem(JobSystem* s) noexcept { m_system = s; }
            JobSystem* system() const noexcept { return m_system; }

            // Route coroutine frame allocation through the per thread pool to
            // keep churn off the global allocator on the hot path.
            static void* operator new(std::size_t size)
            {
                return CoroutineFramePool::allocate(size);
            }
            static void operator delete(void* ptr) noexcept
            {
                CoroutineFramePool::deallocate(ptr);
            }

        protected:
            std::coroutine_handle<> m_continuation {};
            JobSystem*              m_system { nullptr };
        };

        template<typename T>
        class TaskPromise;

        // Define the value promise as a constrained partial specialization.
        // This is more than documentation: it prevents editor frontends from
        // speculatively instantiating its T&& result and aligned storage with
        // T=void before they discover the explicit void specialization below.
        template<NonVoid T>
        class TaskPromise<T> final
            : public TaskPromiseBase
            , public TaskReturnObject<TaskPromise<T>, T>
        {
        public:
            void unhandled_exception() noexcept
            {
                m_exception = std::current_exception();
            }

            template<typename U = T>
                requires std::convertible_to<U&&, T>
            void return_value(U&& value)
                noexcept(std::is_nothrow_constructible_v<T, U&&>)
            {
                ::new (static_cast<void*>(std::addressof(m_storage)))
                    T(std::forward<U>(value));
                m_hasValue = true;
            }

            // Consume the result on the resuming side. Rethrows a captured
            // exception so it propagates to the awaiting coroutine.
            T&& result()
            {
                if (m_exception)
                {
                    std::rethrow_exception(m_exception);
                }
                return std::move(*std::launder(reinterpret_cast<T*>(&m_storage)));
            }

            ~TaskPromise()
            {
                if (m_hasValue)
                {
                    std::destroy_at(
                        std::launder(reinterpret_cast<T*>(&m_storage)));
                }
            }

        private:
            alignas(T) unsigned char m_storage[sizeof(T)];
            bool                     m_hasValue { false };
            std::exception_ptr       m_exception {};
        };

        template<>
        class TaskPromise<void> final
            : public TaskPromiseBase
            , public TaskReturnObject<TaskPromise<void>, void>
        {
        public:
            void unhandled_exception() noexcept
            {
                m_exception = std::current_exception();
            }

            void return_void() noexcept {}

            void result()
            {
                if (m_exception)
                {
                    std::rethrow_exception(m_exception);
                }
            }

        private:
            std::exception_ptr m_exception {};
        };

        // Awaiter for `co_await task`. Hoisted to namespace scope (rather than a
        // local class inside operator co_await) because it needs a member
        // template await_suspend — the caller's promise type is not known until
        // the await site — and local classes may not have member templates.
        template<typename T>
        struct TaskAwaiter
        {
            std::coroutine_handle<TaskPromise<T>> child;

            bool await_ready() const noexcept
            {
                return !child || child.done();
            }

            template<typename CallerPromise>
            std::coroutine_handle<> await_suspend(
                std::coroutine_handle<CallerPromise> caller) noexcept
            {
                ZoneScopedN("Task::await_suspend");

                // Wire the caller as the child's continuation and propagate the
                // job system so the child's final_suspend knows where to
                // reschedule us.
                child.promise().setContinuation(caller);
                child.promise().setSystem(caller.promise().system());

                // Symmetric transfer: start the child now, on this worker,
                // without a nested resume() call — no stack growth.
                return child;
            }

            T await_resume()
            {
                return child.promise().result();
            }
        };
    } // namespace detail


    // Task<T>
    template<typename T>
    class [[nodiscard]] Task
    {
    public:
        using promise_type = detail::TaskPromise<T>;
        using handle_type  = std::coroutine_handle<promise_type>;

        Task() noexcept = default;

        explicit Task(handle_type handle) noexcept : m_handle(handle) {}

        Task(Task&& other) noexcept
            : m_handle(std::exchange(other.m_handle, {}))
        {
        }

        Task& operator=(Task&& other) noexcept
        {
            if (this != &other)
            {
                destroy();
                m_handle = std::exchange(other.m_handle, {});
            }
            return *this;
        }

        Task(const Task&)            = delete;
        Task& operator=(const Task&) = delete;

        ~Task() { destroy(); }

        bool valid() const noexcept { return static_cast<bool>(m_handle); }
        bool done()  const noexcept { return !m_handle || m_handle.done(); }

        handle_type handle() const noexcept { return m_handle; }

        // Hand ownership of the frame to a scheduler (e.g. when_all) that will
        // drive and destroy it explicitly. The Task no longer owns it.
        handle_type release() noexcept { return std::exchange(m_handle, {}); }

        void setSystem(JobSystem* system) noexcept
        {
            if (m_handle)
            {
                m_handle.promise().setSystem(system);
            }
        }

        // co_await a task: suspend the caller, start the child via symmetric
        // transfer, and resume the caller (through the job queue) once the child
        // completes. See detail::TaskAwaiter.
        detail::TaskAwaiter<T> operator co_await() noexcept
        {
            return detail::TaskAwaiter<T>{ m_handle };
        }

    private:
        void destroy() noexcept
        {
            if (m_handle)
            {
                m_handle.destroy();
            }
        }

        handle_type m_handle {};
    };

    namespace detail
    {
        template<typename Promise, typename T>
        Task<T> TaskReturnObject<Promise, T>::get_return_object() noexcept
        {
            auto& promise = static_cast<Promise&>(*this);
            return Task<T>{
                std::coroutine_handle<Promise>::from_promise(promise) };
        }
    } // namespace detail


    // co_await a JobCounter — the bridge that replaces spin-wait for callers
    // that are coroutines.

    // Suspends the awaiting coroutine until the counter reaches zero, at which
    // point the counter's decrement schedules the resume onto the job queue

    
    class CounterAwaiter
    {
    public:
        CounterAwaiter(JobSystem& system, JobCounter& counter) noexcept
            : m_counter(&counter)
        {
            m_node.system = &system;
        }

        bool await_ready() const noexcept
        {
            return m_counter->done();
        }

        bool await_suspend(std::coroutine_handle<> caller) noexcept
        {
            ZoneScopedN("CounterAwaiter::await_suspend");

            m_node.handle = caller;
            m_node.next   = nullptr;

            // Returns false if the counter already closed (resume inline, node
            // not published — safe to keep using the awaiter). Returns true if
            // published — after which we must not touch anything in this frame.
            return m_counter->pushWaiter(&m_node);
        }

        void await_resume() const noexcept {}

    private:
        JobCounter*     m_counter;
        CounterWaitNode m_node {};
    };

    // Awaitable factory: `co_await awaitCounter(jobSystem, counter);`
    inline CounterAwaiter awaitCounter(JobSystem& system, JobCounter& counter) noexcept
    {
        return CounterAwaiter{ system, counter };
    }

    struct JobCounterTaskFactory
    {
        Task<void> operator()(
            JobSystem& system,
            JobCounter& counter) const
        {
            co_await awaitCounter(system, counter);
        }
    };

    inline constexpr JobCounterTaskFactory waitForCounterAsync{};


    // sync_wait — the boundary from the coroutine world back to a blocking
    // (non-coroutine) caller such as the main thread.

    // Drives a Task to completion and returns its result. The calling thread is
    // allowed to block here (it explicitly asked to wait); while it waits it
    // also pumps the job queue so a non-worker thread contributes work and can
    // never deadlock waiting on a resume it is itself responsible for draining —
    // exactly mirroring waitForCounter's main-thread work-stealing.
    namespace detail
    {
        // Caller-owned completion state for sync_wait. Lives on the waiting
        // thread's stack; the runner coroutine only holds a pointer to it.
        struct SyncWaitState
        {
            std::atomic<bool>  done { false };
            std::exception_ptr err {};
        };

        // A dedicated runner coroutine for sync_wait.
        struct SyncWaitTask
        {
            struct promise_type : TaskPromiseBase
            {
                SyncWaitState* state { nullptr };

                struct FinalSignal
                {
                    bool await_ready() const noexcept { return false; }
                    void await_suspend(
                        std::coroutine_handle<promise_type> self) const noexcept
                    {
                        // Read the state pointer out of the frame FIRST, then
                        // publish. Nothing below may touch the frame.
                        SyncWaitState* waitState = self.promise().state;
                        waitState->done.store(true, std::memory_order_release);
                    }
                    void await_resume() const noexcept {}
                };

                SyncWaitTask get_return_object() noexcept
                {
                    return SyncWaitTask{
                        std::coroutine_handle<promise_type>::from_promise(*this) };
                }
                // initial_suspend (suspend_always) inherited from TaskPromiseBase;
                // operator new/delete (frame pool) inherited too.
                FinalSignal final_suspend() noexcept { return {}; }
                void        return_void() noexcept {}
                void        unhandled_exception() noexcept
                {
                    state->err = std::current_exception();
                }
            };

            using handle_type = std::coroutine_handle<promise_type>;
            handle_type handle {};

            explicit SyncWaitTask(handle_type h) noexcept : handle(h) {}
            SyncWaitTask(SyncWaitTask&& o) noexcept
                : handle(std::exchange(o.handle, {})) {}
            SyncWaitTask(const SyncWaitTask&)            = delete;
            SyncWaitTask& operator=(const SyncWaitTask&) = delete;
            ~SyncWaitTask() { if (handle) handle.destroy(); }
        };

        template<typename T>
        SyncWaitTask makeSyncWaitTask(Task<T> task, std::optional<T>& out)
        {
            // Exceptions propagate to promise::unhandled_exception (which records
            // them into SyncWaitState before final_suspend runs).
            out.emplace(co_await std::move(task));
        }

        inline void syncWaitDrive(JobSystem& system,
                                  SyncWaitTask& runner,
                                  SyncWaitState& state)
        {
            runner.handle.promise().state = &state;
            runner.handle.promise().setSystem(&system); // nested awaits fork onto pool
            runner.handle.resume();                      // begin on the calling thread

            while (!state.done.load(std::memory_order_acquire))
            {
                const uint64_t epoch = system.workEpoch();
                if (!system.tryRunOne() &&
                    !state.done.load(std::memory_order_acquire))
                {
                    // Rechecking after sampling the epoch closes the enqueue
                    // race. The atomic wait sleeps until work (normally the
                    // continuation resume) is published.
                    system.waitForWork(epoch);
                }
            }
        }
    } // namespace detail

    template<NonVoid T>
    T sync_wait(JobSystem& system, Task<T> task)
    {
        ZoneScopedN("sync_wait");

        std::optional<T>       out;
        detail::SyncWaitState  state;
        detail::SyncWaitTask   runner =
            detail::makeSyncWaitTask<T>(std::move(task), out);

        detail::syncWaitDrive(system, runner, state);

        if (state.err) std::rethrow_exception(state.err);
        return std::move(*out);
    }

    inline void sync_wait(JobSystem& system, Task<void> task)
    {
        ZoneScopedN("sync_wait");

        detail::SyncWaitState state;

        auto makeRunner = [](Task<void> child) -> detail::SyncWaitTask
        {
            co_await std::move(child);
        };
        detail::SyncWaitTask runner = makeRunner(std::move(task));

        detail::syncWaitDrive(system, runner, state);

        if (state.err) std::rethrow_exception(state.err);
    }

    // when_all — structured fork/join over multiple tasks.
    namespace detail
    {
        // Shared join counter for a when_all group. Initialised to N+1: the
        // extra count is consumed by the when_all coroutine itself when it
        // suspends, which closes the race where every child finishes before the
        // group has actually suspended.
        class WhenAllCounter
        {
        public:
            explicit WhenAllCounter(std::size_t count) noexcept
                : m_count(count + 1)
            {
            }

            // Called by the group coroutine when it suspends. Returns true if it
            // should stay suspended (children still outstanding).
            bool tryAwait(std::coroutine_handle<> continuation, JobSystem* system) noexcept
            {
                m_continuation = continuation;
                m_system       = system;
                return m_count.fetch_sub(1, std::memory_order_acq_rel) > 1;
            }

            // Called by each child on completion; the last one reschedules the
            // group continuation onto the job queue.
            void notify() noexcept
            {
                if (m_count.fetch_sub(1, std::memory_order_acq_rel) == 1)
                {
                    if (m_system)
                    {
                        m_system->scheduleResume(m_continuation);
                    }
                    else if (m_continuation)
                    {
                        m_continuation.resume();
                    }
                }
            }

        private:
            std::atomic<std::size_t> m_count;
            std::coroutine_handle<>  m_continuation {};
            JobSystem*               m_system { nullptr };
        };

        // Wrapper coroutine: runs one child task, stores its result, then
        // notifies the join counter at final_suspend.
        template<typename T>
        class WhenAllTask
        {
        public:
            struct promise_type;
            using handle_type = std::coroutine_handle<promise_type>;

            struct FinalNotify
            {
                bool await_ready() const noexcept { return false; }
                void await_suspend(handle_type h) const noexcept
                {
                    // Notify after the result is stored (return_value already
                    // ran). The wrapper parks here so its stored result stays
                    // alive for the group to read.
                    h.promise().m_counter->notify();
                }
                void await_resume() const noexcept {}
            };

            struct promise_type : TaskPromiseBase
            {
                WhenAllCounter*    m_counter { nullptr };
                std::exception_ptr m_exception {};
                alignas(T) unsigned char m_storage[sizeof(T)];
                bool               m_hasValue { false };

                WhenAllTask get_return_object() noexcept
                {
                    return WhenAllTask{ handle_type::from_promise(*this) };
                }
                std::suspend_always initial_suspend() noexcept { return {}; }
                FinalNotify         final_suspend() noexcept { return {}; }
                void unhandled_exception() noexcept { m_exception = std::current_exception(); }

                template<typename U>
                void return_value(U&& value)
                {
                    ::new (static_cast<void*>(std::addressof(m_storage)))
                        T(std::forward<U>(value));
                    m_hasValue = true;
                }

                ~promise_type()
                {
                    if (m_hasValue)
                    {
                        std::destroy_at(
                            std::launder(reinterpret_cast<T*>(&m_storage)));
                    }
                }
            };

            explicit WhenAllTask(handle_type h) noexcept : m_handle(h) {}
            WhenAllTask(WhenAllTask&& o) noexcept : m_handle(std::exchange(o.m_handle, {})) {}
            WhenAllTask(const WhenAllTask&) = delete;
            WhenAllTask& operator=(const WhenAllTask&) = delete;
            ~WhenAllTask() { if (m_handle) m_handle.destroy(); }

            void start(WhenAllCounter& counter, JobSystem& system) noexcept
            {
                m_handle.promise().m_counter = &counter;
                m_handle.promise().setSystem(&system);
                // Fork: schedule the child onto the pool so groups run in
                // parallel across workers.
                system.scheduleResume(m_handle);
            }

            T&& result()
            {
                auto& p = m_handle.promise();
                if (p.m_exception)
                {
                    std::rethrow_exception(p.m_exception);
                }
                return std::move(*std::launder(reinterpret_cast<T*>(&p.m_storage)));
            }

        private:
            handle_type m_handle;
        };

        // void specialisation.
        template<>
        class WhenAllTask<void>
        {
        public:
            struct promise_type;
            using handle_type = std::coroutine_handle<promise_type>;

            struct FinalNotify
            {
                bool await_ready() const noexcept { return false; }
                void await_suspend(handle_type h) const noexcept
                {
                    h.promise().m_counter->notify();
                }
                void await_resume() const noexcept {}
            };

            struct promise_type : TaskPromiseBase
            {
                WhenAllCounter*    m_counter { nullptr };
                std::exception_ptr m_exception {};

                WhenAllTask get_return_object() noexcept
                {
                    return WhenAllTask{ handle_type::from_promise(*this) };
                }
                std::suspend_always initial_suspend() noexcept { return {}; }
                FinalNotify         final_suspend() noexcept { return {}; }
                void unhandled_exception() noexcept { m_exception = std::current_exception(); }
                void return_void() noexcept {}
            };

            explicit WhenAllTask(handle_type h) noexcept : m_handle(h) {}
            WhenAllTask(WhenAllTask&& o) noexcept : m_handle(std::exchange(o.m_handle, {})) {}
            WhenAllTask(const WhenAllTask&) = delete;
            WhenAllTask& operator=(const WhenAllTask&) = delete;
            ~WhenAllTask() { if (m_handle) m_handle.destroy(); }

            void start(WhenAllCounter& counter, JobSystem& system) noexcept
            {
                m_handle.promise().m_counter = &counter;
                m_handle.promise().setSystem(&system);
                system.scheduleResume(m_handle);
            }

            void result()
            {
                auto& p = m_handle.promise();
                if (p.m_exception)
                {
                    std::rethrow_exception(p.m_exception);
                }
            }

        private:
            handle_type m_handle;
        };

        // Adapt a Task<T> into a WhenAllTask<T> that feeds the join counter.
        template<typename T>
        WhenAllTask<T> makeWhenAllTask(Task<T> task)
        {
            co_return co_await std::move(task);
        }

        template<>
        inline WhenAllTask<void> makeWhenAllTask<void>(Task<void> task)
        {
            co_await std::move(task);
        }

        // Awaitable that suspends the group coroutine on the join counter.
        struct WhenAllAwaitable
        {
            WhenAllCounter& counter;

            bool await_ready() const noexcept { return false; }

            template<typename Promise>
            bool await_suspend(std::coroutine_handle<Promise> group) noexcept
            {
                return counter.tryAwait(group, group.promise().system());
            }

            void await_resume() const noexcept {}
        };
    } // namespace detail

    // Variadic when_all: runs every task in parallel across the pool and
    // completes when all have finished, yielding a tuple of results (void
    // results become std::monostate to keep the tuple well formed).
    template<typename... Ts>
    Task<std::tuple<std::conditional_t<std::is_void_v<Ts>, std::monostate, Ts>...>>
    when_all(JobSystem& system, Task<Ts>... tasks)
    {
        detail::WhenAllCounter counter{ sizeof...(Ts) };

        // Wrap each task and fork it onto the pool.
        auto wrappers = std::make_tuple(detail::makeWhenAllTask<Ts>(std::move(tasks))...);

        std::apply(
            [&](auto&... wrapper) { (wrapper.start(counter, system), ...); },
            wrappers);

        // Suspend the group until the join counter drains, then resume via queue.
        co_await detail::WhenAllAwaitable{ counter };

        // All children complete; gather results.
        co_return std::apply(
            [](auto&... wrapper)
            {
                auto gather = [](auto& w) -> decltype(auto)
                {
                    using WT = std::remove_reference_t<decltype(w)>;
                    if constexpr (std::is_same_v<WT, detail::WhenAllTask<void>>)
                    {
                        w.result();
                        return std::monostate{};
                    }
                    else
                    {
                        return w.result();
                    }
                };
                return std::make_tuple(gather(wrapper)...);
            },
            wrappers);
    }

    // Homogeneous vector fork/join: "N parallel jobs"
    template<typename T>
    Task<std::vector<T>> when_all(JobSystem& system, std::vector<Task<T>> tasks)
    {
        detail::WhenAllCounter counter{ tasks.size() };

        std::vector<detail::WhenAllTask<T>> wrappers;
        wrappers.reserve(tasks.size());
        for (auto& t : tasks)
        {
            wrappers.push_back(detail::makeWhenAllTask<T>(std::move(t)));
        }
        for (auto& w : wrappers)
        {
            w.start(counter, system);
        }

        co_await detail::WhenAllAwaitable{ counter };

        std::vector<T> results;
        results.reserve(wrappers.size());
        for (auto& w : wrappers)
        {
            results.push_back(w.result());
        }
        co_return results;
    }

    // void vector fork/join: a pure barrier over N tasks.
    inline Task<void> when_all(JobSystem& system, std::vector<Task<void>> tasks)
    {
        detail::WhenAllCounter counter{ tasks.size() };

        std::vector<detail::WhenAllTask<void>> wrappers;
        wrappers.reserve(tasks.size());
        for (auto& t : tasks)
        {
            wrappers.push_back(detail::makeWhenAllTask<void>(std::move(t)));
        }
        for (auto& w : wrappers)
        {
            w.start(counter, system);
        }

        co_await detail::WhenAllAwaitable{ counter };

        for (auto& w : wrappers)
        {
            w.result(); // propagate any exception
        }
        co_return;
    }
}
