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
            , m_resourceAccesses(memory)
            , m_barriers(memory)
            , m_resourceLifetimes(memory)
            , m_dependencies(memory)
            , m_resourceChains(memory)
            , m_chainMap(memory)
            , m_adjacencyOffsets(memory)
            , m_adjacencyCounts(memory)
            , m_adjacencyEdges(memory)
            , m_nodeSchedules(memory)
            , m_executionLevels(memory)
            , m_executionLevelNodes(memory)
            , m_queueSubmissions(memory)
            , m_queueSubmissionNodes(memory)
            , m_queueSubmissionWaits(memory)
            , m_incomingBarrierIndices(memory)
            , m_outgoingBarrierIndices(memory)
        {}

        CompiledGraphPlan compile(
            const FrameGraphBuilder& builder);

        void buildDependencies();

        void buildExecutionOrder();
        void buildExecutionLevels();
        void buildQueueSubmissions();

        void buildResourceLifetimes();

        void buildResourceAccessChains(const FrameGraphBuilder& builder);

        void buildAdjacencyLists(); 

        void buildBarriers();

        void buildNodeSchedules();

    private:

        void debugLog();

        std::pmr::vector<ExecutionNode> m_nodes;
        std::pmr::vector<ResourceAccess> m_resourceAccesses;
        std::pmr::vector<ResourceBarrier> m_barriers;
        std::pmr::vector<ResourceLifetime> m_resourceLifetimes;
        std::pmr::vector<Dependency> m_dependencies;
        std::pmr::vector<GraphNodeId> m_executionOrder;
        std::pmr::unordered_map<GraphResourceId, ResourceChain> m_chainMap;
        std::pmr::vector<ResourceChain> m_resourceChains;
        std::pmr::vector<uint32_t> m_adjacencyOffsets;
        std::pmr::vector<uint32_t> m_adjacencyCounts;
        std::pmr::vector<GraphNodeId> m_adjacencyEdges;
        std::pmr::vector<NodeSchedule> m_nodeSchedules;
        std::pmr::vector<ExecutionLevel> m_executionLevels;
        std::pmr::vector<GraphNodeId> m_executionLevelNodes;
        std::pmr::vector<QueueSubmissionBatch> m_queueSubmissions;
        std::pmr::vector<GraphNodeId> m_queueSubmissionNodes;
        std::pmr::vector<QueueSubmissionWait> m_queueSubmissionWaits;
        std::pmr::vector<uint32_t> m_incomingBarrierIndices;
        std::pmr::vector<uint32_t> m_outgoingBarrierIndices;
    };
}
