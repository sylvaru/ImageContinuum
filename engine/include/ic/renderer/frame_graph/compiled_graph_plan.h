// ic/renderer/frame_graph/compiled_graph_plan.h
#pragma once
#include <span>
#include "frame_graph_pass.h"
#include "frame_graph_types.h"

namespace ic
{


    struct CompiledGraphPlan
    {
        std::span<const ExecutionNode> nodes;

        std::span<const GraphNodeId> executionOrder;

        std::span<const std::pmr::vector<GraphNodeId>> executionLevels;

        std::span<const Dependency> dependencies;

        std::span<const ResourceBarrier> barriers;

        std::span<const NodeSchedule> nodeSchedules;

        std::span<const ResourceLifetime> resourceLifetimes;

        std::span<const GraphResource> resources;

        std::span<const PassPayload> payloads;
    };

}
