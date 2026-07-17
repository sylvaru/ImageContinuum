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

        // Minimal cross-frame GPU ordering edges (see CrossFrameDependency).
        // Empty means no resource's physical memory is reused across frames with
        // a hazard, so consecutive frames may overlap fully on the GPU.
        std::span<const CrossFrameDependency> crossFrameDependencies;
    };

    struct GraphExecutionContext
    {
        // Empty means execute every node, preserving compatibility for tools
        // and callers that do not use frequency scheduling.
        std::span<const uint8_t> executeNodes;

        [[nodiscard]] bool shouldExecute(GraphNodeId node) const noexcept
        {
            return executeNodes.empty() ||
                (node < executeNodes.size() && executeNodes[node] != 0);
        }
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

    [[nodiscard]] inline GraphResourceId findPreviousNodeResource(
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
            if (access.previousVersion && access.usage == usage)
            {
                return access.resource;
            }
        }
        return InvalidGraphResourceId;
    }

    // Resolves the single graph resource carrying the given semantic. Backends
    // use this to bind the frame-graph-owned (registry) resource instead of a
    // duplicate backend-managed buffer, so the graph's barrier/queue machinery
    // owns that resource's state, transitions and lifetime. Semantics are unique
    // per graph (set via FrameGraphBuilder::setResourceSemantic), so a global
    // lookup is correct and independent of which pass is being recorded.
    [[nodiscard]] inline GraphResourceId findResourceBySemantic(
        const CompiledGraphPlan& plan,
        GraphResourceSemantic semantic) noexcept
    {
        for (GraphResourceId id = 0; id < plan.resources.size(); ++id)
        {
            if (plan.resources[id].semantic == semantic)
            {
                return id;
            }
        }
        return InvalidGraphResourceId;
    }

    [[nodiscard]] inline bool nodeUsesResourceSemantic(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node,
        GraphResourceSemantic semantic) noexcept
    {
        if (node.firstResourceAccess > plan.resourceAccesses.size() ||
            node.resourceAccessCount >
                plan.resourceAccesses.size() - node.firstResourceAccess)
        {
            return false;
        }
        for (const ResourceAccess& access : plan.resourceAccesses.subspan(
                 node.firstResourceAccess, node.resourceAccessCount))
        {
            if (access.resource < plan.resources.size() &&
                plan.resources[access.resource].semantic == semantic)
            {
                return true;
            }
        }
        return false;
    }

}
