// ic/renderer/frame_graph/compiled_graph_plan.h
#pragma once
#include <cstddef>
#include <span>
#include "frame_graph_pass.h"
#include "frame_graph_types.h"

namespace ic
{


    struct CompiledGraphPlan
    {
        std::span<const ExecutionNode> nodes;

        std::span<const GraphNodeId> executionOrder;

        std::span<const ExecutionLevel> executionLevels;

        std::span<const GraphNodeId> executionLevelNodes;

        std::span<const QueueSubmissionBatch> queueSubmissions;
        std::span<const GraphNodeId> queueSubmissionNodes;
        std::span<const QueueSubmissionWait> queueSubmissionWaits;

        std::span<const Dependency> dependencies;

        std::span<const ResourceBarrier> barriers;

        std::span<const NodeSchedule> nodeSchedules;

        std::span<const uint32_t> incomingBarrierIndices;

        std::span<const uint32_t> outgoingBarrierIndices;

        std::span<const ResourceLifetime> resourceLifetimes;

        std::span<const ResourceAccess> resourceAccesses;

        std::span<const GraphResource> resources;

        std::span<const PassPayload> payloads;
    };

    [[nodiscard]] inline GraphResourceId findNodeResource(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node,
        ResourceUsage usage) noexcept
    {
        const std::size_t first = node.firstResourceAccess;
        const std::size_t count = node.resourceAccessCount;
        if (first > plan.resourceAccesses.size() ||
            count > plan.resourceAccesses.size() - first)
        {
            return InvalidGraphResourceId;
        }

        for (const ResourceAccess& access :
             plan.resourceAccesses.subspan(first, count))
        {
            if (access.usage == usage)
            {
                return access.resource;
            }
        }
        return InvalidGraphResourceId;
    }

}
