// ic/renderer/frame_graph/frame_graph_types.h
#pragma once
#include <span>

namespace ic
{
    struct CompiledNode;
    struct Barrier;
    struct ResourceLifetime;

    struct CompiledFramePlan
    {
        std::span<const CompiledNode> nodes;
        std::span<const Barrier> barriers;
        std::span<const ResourceLifetime> resources;
    };

}