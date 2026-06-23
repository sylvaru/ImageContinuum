// ic/rendering/frame_graph_node.h
#pragma once
#include "ic/renderer/render_types.h"

namespace ic 
{
    class FrameGraphBuilder;
    class RenderContext;

    // The Concept enforces that a pass must define how it connects to the DAG
    template<typename T>
    concept FrameGraphNode = requires(T node, FrameGraphBuilder& builder, const RenderContext& ctx)
    {
        // Setup reads/writes handles to explicitly map the DAG's edges
        { node.setup(builder) } -> std::same_as<void>;

        // Execute consumes an agnostic context to record commands
        { node.execute(ctx) } -> std::same_as<void>;
    };
}