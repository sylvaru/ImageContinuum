// ic/renderer/frame_graph/compiled_graph_plan.h
#pragma once
#include <span>

namespace ic
{
    struct ExecutionNode;
    struct Barrier;
    struct ResourceLifetime;
    struct Dependency;


    struct CompiledGraphPlan
    {
        std::span<const ExecutionNode> nodes;

        std::span<const Dependency> dependencies;

        std::span<const Barrier> barriers;

        std::span<const ResourceLifetime> resourceLifetimes;
    };

}
