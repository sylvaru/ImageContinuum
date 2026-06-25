#pragma once
#include "compiled_graph_plan.h"

namespace ic
{
    class FrameGraphBuilder;

    class FrameGraphCompiler
    {
    public:
        explicit FrameGraphCompiler(
            std::pmr::memory_resource* memory)
            : m_nodes(memory)
            , m_barriers(memory)
            , m_resourceLifetimes(memory)
            , m_dependencies(memory)
        {}

        CompiledGraphPlan compile(
            const FrameGraphBuilder& builder);
    private:
        std::pmr::vector<ExecutionNode> m_nodes;
        std::pmr::vector<Barrier> m_barriers;
        std::pmr::vector<ResourceLifetime> m_resourceLifetimes;
        std::pmr::vector<Dependency> m_dependencies;
    };
}