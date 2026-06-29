#pragma once
#include "compiled_graph_plan.h"
#include "frame_graph_types.h"

namespace ic
{
    class FrameGraphBuilder;

    struct ResourceChain
    {
        GraphResourceId resource;
        std::pmr::vector<ResourceAccess> accesses;

        ResourceChain(
            GraphResourceId r,
            std::pmr::memory_resource* alloc)
            : resource(r)
            , accesses(alloc)
        {
        }
    };

    class FrameGraphCompiler
    {
    public:
        explicit FrameGraphCompiler(
            std::pmr::memory_resource* memory)
            : m_nodes(memory)
            , m_barriers(memory)
            , m_resourceLifetimes(memory)
            , m_dependencies(memory)
            , m_resourceChains(memory)
            , m_chainMap(memory)
            , m_adjacencyLists(memory)
            , m_nodeSchedules(memory)
            , m_executionLevels(memory)
        {}

        CompiledGraphPlan compile(
            const FrameGraphBuilder& builder);

        void buildDependencies();

        void buildExecutionOrder();
        void buildExecutionLevels();

        void buildResourceLifetimes();

        void buildResourceAccessChains(const FrameGraphBuilder& builder);

        void buildAdjacencyLists(); 

        void buildBarriers();

        void buildNodeSchedules();

    private:

        void debugLog();

        std::pmr::vector<ExecutionNode> m_nodes;
        std::pmr::vector<ResourceBarrier> m_barriers;
        std::pmr::vector<ResourceLifetime> m_resourceLifetimes;
        std::pmr::vector<Dependency> m_dependencies;
        std::pmr::vector<GraphNodeId> m_executionOrder;
        std::pmr::unordered_map<GraphResourceId, ResourceChain> m_chainMap;
        std::pmr::vector<ResourceChain> m_resourceChains;
        std::pmr::vector<std::pmr::vector<GraphNodeId>> m_adjacencyLists;
        std::pmr::vector<NodeSchedule> m_nodeSchedules;
        std::pmr::vector<std::pmr::vector<GraphNodeId>> m_executionLevels;
    };
}
