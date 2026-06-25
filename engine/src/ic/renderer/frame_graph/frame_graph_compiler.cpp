#include "ic/common/ic_pch.h"
#include "ic/renderer/frame_graph/frame_graph_compiler.h"
#include "ic/renderer/frame_graph/frame_graph_builder.h"
#include "ic/renderer/frame_graph/frame_graph_types.h"


namespace ic
{
    CompiledGraphPlan
        FrameGraphCompiler::compile(
            const FrameGraphBuilder& builder)
    {
        m_nodes.clear();
        m_barriers.clear();
        m_resourceLifetimes.clear();

        auto nodes =
            builder.nodes();

        for (const auto& node : nodes)
        {
            ExecutionNode compiledNode{};

            compiledNode.nodeId =
                node.id;

            compiledNode.queue =
                node.queue;

            compiledNode.type =
                node.type;

            m_nodes.push_back(
                compiledNode);
        }

        return {
            .nodes =
                std::span<const ExecutionNode>(
                    m_nodes),

            .barriers =
                std::span<const Barrier>(
                    m_barriers),

            .resourceLifetimes =
                std::span<const ResourceLifetime>(
                    m_resourceLifetimes)
        };
    }
}