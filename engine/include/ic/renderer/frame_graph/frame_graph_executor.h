#pragma once

#include "compiled_graph_plan.h"
#include "ic/core/job_system.h"
#include "ic/core/task.h"

#include <algorithm>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <spdlog/spdlog.h>
#include <vector>

namespace ic
{
    namespace detail
    {
        // Coroutine driver for parallel frame-graph recording.
        //
        // Each execution level still fans its independent nodes out onto the
        // worker pool through the existing, allocation free JobTask path (no
        // coroutine frame per pass). What changes is the level barrier: instead
        // of spinning the calling thread on the counter with _mm_pause
        // (waitForCounter), we `co_await` the level's JobCounter. The driver
        // suspends and is resumed — via the job queue — the instant the last
        // node in the level finishes recording. Consecutive levels are
        // sequential co_awaits, so level boundaries remain hard synchronization
        // points and output ordering stays deterministic (matches
        // executionLevelNodes).
        //
        // recordNode/output/plan are passed by reference; their referents are
        // locals of the synchronous recordFrameGraph frame, which stays alive
        // across the sync_wait that drives this coroutine. The per level
        // exception state lives in this coroutine frame, which is suspended (not
        // destroyed) at each co_await, so the kicked JobTasks may safely
        // reference it while they run.
        template<typename Command, typename RecordNode>
        Task<void> recordFrameGraphCoroutine(
            const CompiledGraphPlan& plan,
            JobSystem& jobs,
            uint32_t slotCount,
            RecordNode& recordNode,
            std::vector<Command>& output)
        {
            std::vector<JobTask> tasks;
            uint32_t largestLevel = 0;
            for (const ExecutionLevel& level : plan.executionLevels)
            {
                largestLevel = std::max(largestLevel, level.nodeCount);
            }
            tasks.reserve(largestLevel);

            for (const ExecutionLevel& level : plan.executionLevels)
            {
                struct FailedRecording
                {
                    GraphNodeId node = InvalidGraphNodeId;
                    uint32_t outputIndex = UINT32_MAX;
                    std::string error;
                };
                std::vector<FailedRecording> failures;
                std::mutex exceptionMutex;
                tasks.clear();

                for (uint32_t i = 0; i < level.nodeCount; ++i)
                {
                    const GraphNodeId node =
                        plan.executionLevelNodes[level.firstNode + i];
                    const uint32_t outputIndex = level.firstNode + i;
                    const uint32_t workerIndex = i % slotCount;

                    tasks.push_back(JobTask::make(
                        [&, node, outputIndex, workerIndex]
                        {
                            try
                            {
                                output[outputIndex] =
                                    recordNode(node, workerIndex);
                            }
                            catch (const std::exception& error)
                            {
                                std::scoped_lock lock(exceptionMutex);
                                failures.push_back({ node, outputIndex, error.what() });
                            }
                            catch (...)
                            {
                                std::scoped_lock lock(exceptionMutex);
                                failures.push_back({ node, outputIndex, "unknown exception" });
                            }
                        }));
                }

                JobCounter counter;
                jobs.kickTasks(
                    tasks.data(),
                    static_cast<uint32_t>(tasks.size()),
                    &counter);

                // Coroutine level barrier: suspend the driver (freeing the
                // thread to help drain other work) until every node in this
                // level has been recorded. No busy spin.
                co_await awaitCounter(jobs, counter);

                for (const FailedRecording& failure : failures)
                {
                    spdlog::warn(
                        "[FrameGraph] Parallel recording failed for node {}: {}. Retrying serially.",
                        failure.node,
                        failure.error);
                    try
                    {
                        output[failure.outputIndex] =
                            recordNode(failure.node, 0);
                    }
                    catch (const std::exception& retryError)
                    {
                        throw std::runtime_error(
                            "Frame-graph recording failed for node " +
                            std::to_string(failure.node) +
                            " after serial retry: " + retryError.what());
                    }
                }
            }

            co_return;
        }
    } // namespace detail

    // Records independent nodes in the same execution level concurrently.
    // Level boundaries remain synchronization points, while output ordering is
    // deterministic and matches executionLevelNodes.
    template<typename Command, typename RecordNode>
    void recordFrameGraph(
        const CompiledGraphPlan& plan,
        JobSystem* jobs,
        uint32_t workerSlots,
        RecordNode&& recordNode,
        std::vector<Command>& output)
    {
        const uint32_t slotCount = std::max(1u, workerSlots);
        if (plan.executionLevels.empty() || !jobs ||
            jobs->workerCount() == 0)
        {
            output.reserve(output.size() + plan.executionOrder.size());
            for (uint32_t i = 0; i < plan.executionOrder.size(); ++i)
            {
                output.push_back(
                    recordNode(plan.executionOrder[i], i % slotCount));
            }
            return;
        }

        output.resize(plan.executionLevelNodes.size());

        // Drive the coroutine from this synchronous (render-thread) call site.
        // sync_wait pumps the job queue while it waits, so the render thread
        // still contributes recording work and cannot deadlock on a resume it is
        // responsible for draining — exactly matching the old work-stealing
        // waitForCounter, minus the spin.
        sync_wait(
            *jobs,
            detail::recordFrameGraphCoroutine<Command>(
                plan, *jobs, slotCount, recordNode, output));
    }
}
