#pragma once

#include "compiled_graph_plan.h"
#include "ic/core/job_system.h"

#include <algorithm>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <spdlog/spdlog.h>
#include <vector>

namespace ic
{
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
        std::mutex exceptionMutex;
        std::vector<JobTask> tasks;
        uint32_t largestLevel = 0;
        for (const ExecutionLevel& level : plan.executionLevels)
        {
            largestLevel = std::max(largestLevel, level.nodeCount);
        }
        tasks.reserve(largestLevel);

        for (const ExecutionLevel& level : plan.executionLevels)
        {
            std::exception_ptr recordingException;
            std::string recordingError;
            GraphNodeId failedNode = InvalidGraphNodeId;
            uint32_t failedOutputIndex = UINT32_MAX;
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
                            if (!recordingException)
                            {
                                recordingException =
                                    std::current_exception();
                                recordingError = error.what();
                                failedNode = node;
                                failedOutputIndex = outputIndex;
                            }
                        }
                        catch (...)
                        {
                            std::scoped_lock lock(exceptionMutex);
                            if (!recordingException)
                            {
                                recordingException =
                                    std::current_exception();
                                recordingError = "unknown exception";
                                failedNode = node;
                                failedOutputIndex = outputIndex;
                            }
                        }
                    }));
            }

            JobCounter counter;
            jobs->kickTasks(
                tasks.data(),
                static_cast<uint32_t>(tasks.size()),
                &counter);
            jobs->waitForCounter(&counter);

            if (recordingException)
            {
                spdlog::warn(
                    "[FrameGraph] Parallel recording failed for node {}: {}. Retrying serially.",
                    failedNode,
                    recordingError);
                try
                {
                    output[failedOutputIndex] =
                        recordNode(failedNode, 0);
                }
                catch (const std::exception& retryError)
                {
                    throw std::runtime_error(
                        "Frame-graph recording failed for node " +
                        std::to_string(failedNode) +
                        " after serial retry: " + retryError.what());
                }
            }
        }
    }
}
