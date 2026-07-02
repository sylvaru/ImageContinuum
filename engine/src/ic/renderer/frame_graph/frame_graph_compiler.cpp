#include "ic/common/ic_pch.h"
#include "ic/renderer/frame_graph/frame_graph_compiler.h"
#include "ic/renderer/frame_graph/frame_graph_builder.h"
#include "ic/renderer/frame_graph/frame_graph_types.h"


#include <queue>
#include <spdlog/spdlog.h>

namespace ic
{
    CompiledGraphPlan
        FrameGraphCompiler::compile(
            const FrameGraphBuilder& builder)
    {
        m_nodes.clear();
        m_barriers.clear();
        m_resourceLifetimes.clear();
        m_dependencies.clear();
        m_executionOrder.clear();
        m_executionLevels.clear();
        m_executionLevelNodes.clear();
        m_chainMap.clear();
        m_resourceChains.clear();
        m_nodeSchedules.clear();
        m_adjacencyOffsets.clear();
        m_adjacencyCounts.clear();
        m_adjacencyEdges.clear();
        m_incomingBarrierIndices.clear();
        m_outgoingBarrierIndices.clear();
  
        auto nodes = builder.nodes();

        // Build node list
        for (const auto& node : nodes)
        {
            ExecutionNode compiledNode{};

            compiledNode.nodeId = node.graphNode.id;
            compiledNode.queue = node.graphNode.queue;
            compiledNode.type = node.graphNode.type;
            compiledNode.payloadIndex = node.graphNode.payloadIndex;

            m_nodes.push_back(compiledNode);
        }

        buildResourceAccessChains(builder);
        buildDependencies();
        buildAdjacencyLists();
        buildExecutionOrder();
        buildExecutionLevels();
        buildResourceLifetimes();
        buildBarriers();
        buildNodeSchedules();

        debugLog();

        auto resources = builder.resources();
        auto payloads = builder.payloads();

        return
        {
            .nodes = std::span<const ExecutionNode>(m_nodes),
            .executionOrder = std::span<const GraphNodeId>(m_executionOrder),
            .executionLevels = std::span<const ExecutionLevel>(m_executionLevels),
            .executionLevelNodes = std::span<const GraphNodeId>(m_executionLevelNodes),
            .dependencies = std::span<const Dependency>(m_dependencies),
            .barriers = std::span<const ResourceBarrier>(m_barriers),
            .nodeSchedules = std::span<const NodeSchedule>(m_nodeSchedules),
            .incomingBarrierIndices = std::span<const uint32_t>(m_incomingBarrierIndices),
            .outgoingBarrierIndices = std::span<const uint32_t>(m_outgoingBarrierIndices),
            .resourceLifetimes = std::span<const ResourceLifetime>(m_resourceLifetimes),
            .resources = std::span<const GraphResource>(resources),
            .payloads = std::span<const PassPayload>(payloads)
        };
    }

    void FrameGraphCompiler::buildDependencies()
    {
        m_dependencies.clear();

        struct LastWriter
        {
            GraphNodeId node{};
            bool valid = false;
        };

        std::pmr::unordered_map<
            GraphResourceId,
            LastWriter> lastWriterMap;

        for (const auto& chain : m_resourceChains)
        {
            LastWriter lastWriter{};

            for (const auto& access : chain.accesses)
            {
                auto& lw = lastWriterMap[chain.resource];

                if (access.access == AccessType::Write)
                {
                    // write-after-write OR write-after-read
                    if (lw.valid && lw.node != access.node)
                    {
                        m_dependencies.push_back({
                            .source = lw.node,
                            .destination = access.node
                            });
                    }

                    lw = { access.node, true };
                }
                else // Read
                {
                    // read-after-write dependency
                    if (lw.valid && lw.node != access.node)
                    {
                        m_dependencies.push_back({
                            .source = lw.node,
                            .destination = access.node
                            });
                    }
                }
            }
        }
    }

    void FrameGraphCompiler::buildAdjacencyLists()
    {
        const size_t nodeCount = m_nodes.size();

        m_adjacencyOffsets.clear();
        m_adjacencyCounts.clear();
        m_adjacencyEdges.clear();

        m_adjacencyOffsets.resize(nodeCount, 0);
        m_adjacencyCounts.resize(nodeCount, 0);
        m_adjacencyEdges.resize(m_dependencies.size());

        for (const auto& dep : m_dependencies)
        {
            ++m_adjacencyCounts[dep.source];
        }

        uint32_t offset = 0;
        for (size_t i = 0; i < nodeCount; ++i)
        {
            m_adjacencyOffsets[i] = offset;
            offset += m_adjacencyCounts[i];
        }

        std::pmr::vector<uint32_t> cursors(
            m_adjacencyOffsets.begin(),
            m_adjacencyOffsets.end(),
            m_nodes.get_allocator());

        for (const auto& dep : m_dependencies)
        {
            const uint32_t index = cursors[dep.source]++;
            m_adjacencyEdges[index] = dep.destination;
        }

        for (size_t node = 0; node < nodeCount; ++node)
        {
            auto begin =
                m_adjacencyEdges.begin() + m_adjacencyOffsets[node];
            auto end = begin + m_adjacencyCounts[node];
            std::sort(begin, end);
        }
    }

    void FrameGraphCompiler::buildExecutionOrder()
    {
        const size_t nodeCount = m_nodes.size();

        std::pmr::vector<uint32_t> inDegree(
            nodeCount, 0, m_nodes.get_allocator()
        );

        // Compute in-degree
        for (const auto& dep : m_dependencies)
        {
            inDegree[dep.destination]++;
        }

        // Deterministic priority queue LOWEST ID FIRST
        std::priority_queue<
            GraphNodeId,
            std::vector<GraphNodeId>,
            std::greater<GraphNodeId>
        > queue;

        // Seed queue with all zero-degree nodes
        for (GraphNodeId i = 0; i < nodeCount; ++i)
        {
            if (inDegree[i] == 0)
                queue.push(i);
        }

        m_executionOrder.clear();
        m_executionOrder.reserve(nodeCount);

        // Kahn processing
        while (!queue.empty())
        {
            GraphNodeId n = queue.top();
            queue.pop();

            m_executionOrder.push_back(n);

            const uint32_t firstEdge = m_adjacencyOffsets[n];
            const uint32_t edgeCount = m_adjacencyCounts[n];

            for (uint32_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex)
            {
                const GraphNodeId m =
                    m_adjacencyEdges[firstEdge + edgeIndex];

                if (--inDegree[m] == 0)
                {
                    queue.push(m);
                }
            }
        }

        // Cycle check
        assert(
            m_executionOrder.size() == nodeCount &&
            "FrameGraph contains dependency cycle"
        );
    }

    void FrameGraphCompiler::buildExecutionLevels()
    {
        const size_t nodeCount = m_nodes.size();

        std::pmr::vector<uint32_t> inDegree(
            nodeCount, 0, m_nodes.get_allocator());

        for (const auto& dep : m_dependencies)
        {
            inDegree[dep.destination]++;
        }

        std::pmr::vector<GraphNodeId> currentLevel(m_nodes.get_allocator());
        currentLevel.reserve(nodeCount);

        for (GraphNodeId node = 0; node < nodeCount; ++node)
        {
            if (inDegree[node] == 0)
            {
                currentLevel.push_back(node);
            }
        }

        m_executionLevels.clear();
        m_executionLevelNodes.clear();

        size_t scheduledCount = 0;

        while (!currentLevel.empty())
        {
            std::sort(currentLevel.begin(), currentLevel.end());

            ExecutionLevel level{};
            level.firstNode =
                static_cast<uint32_t>(m_executionLevelNodes.size());
            level.nodeCount =
                static_cast<uint32_t>(currentLevel.size());

            m_executionLevelNodes.insert(
                m_executionLevelNodes.end(),
                currentLevel.begin(),
                currentLevel.end());

            scheduledCount += currentLevel.size();
            m_executionLevels.push_back(level);

            std::pmr::vector<GraphNodeId> nextLevel(m_nodes.get_allocator());

            for (GraphNodeId node : currentLevel)
            {
                const uint32_t firstEdge = m_adjacencyOffsets[node];
                const uint32_t edgeCount = m_adjacencyCounts[node];

                for (uint32_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex)
                {
                    const GraphNodeId destination =
                        m_adjacencyEdges[firstEdge + edgeIndex];

                    if (--inDegree[destination] == 0)
                    {
                        nextLevel.push_back(destination);
                    }
                }
            }

            currentLevel = std::move(nextLevel);
        }

        assert(
            scheduledCount == nodeCount &&
            "FrameGraph contains dependency cycle");
    }

    void FrameGraphCompiler::buildResourceLifetimes()
    {
        m_resourceLifetimes.clear();

        std::pmr::vector<uint32_t> execIndex(
            m_nodes.size(),
            0,
            m_nodes.get_allocator()
        );

        for (uint32_t i = 0; i < m_executionOrder.size(); ++i)
        {
            execIndex[m_executionOrder[i]] = i;
        }

        for (auto& chain : m_resourceChains)
        {
            if (chain.accesses.empty())
                continue;

            auto sorted = chain.accesses;

            std::sort(sorted.begin(), sorted.end(),
                [&](const ResourceAccess& a, const ResourceAccess& b)
                {
                    const bool aExternal = a.external;
                    const bool bExternal = b.external;

                    if (aExternal != bExternal)
                        return aExternal;

                    return execIndex[a.node] < execIndex[b.node];
                });

            m_resourceLifetimes.push_back({
                .resource = chain.resource,
                .firstUse = sorted.front().node,
                .lastUse = sorted.back().node
                });
        }
    }

    void FrameGraphCompiler::buildResourceAccessChains(
        const FrameGraphBuilder& builder)
    {
        m_resourceChains.clear();
        m_chainMap.clear();

        for (const ResourceAccess& access : builder.accesses())
        {
            auto& chain =
                m_chainMap.try_emplace(
                    access.resource,
                    access.resource,
                    builder.get_allocator()
                ).first->second;

            chain.accesses.push_back(access);
        }

        m_resourceChains.clear();
        m_resourceChains.reserve(m_chainMap.size());

        for (auto& [id, chain] : m_chainMap)
        {
            m_resourceChains.push_back(std::move(chain));
        }

        auto resources = builder.resources();

        for (auto& chain : m_resourceChains)
        {
            const auto& resource = resources[chain.resource];

            if (resource.ownership != ResourceOwnership::Imported)
                continue;

            if (chain.accesses.empty())
                continue;


            if (resource.ownership == ResourceOwnership::Imported)
            {
                ResourceAccess initial{};
                initial.node = chain.accesses.front().node;
                initial.resource = chain.resource;

                initial.access = resource.initialAccess;
                initial.usage = resource.initialUsage;
                initial.external = true;
                initial.firstUse = true;

                chain.accesses.insert(chain.accesses.begin(), initial);
            }
        }
    }

    void FrameGraphCompiler::buildBarriers()
    {
        m_barriers.clear();

        std::pmr::vector<uint32_t> execIndex(
            m_nodes.size(),
            0,
            m_nodes.get_allocator()
        );

        for (uint32_t i = 0; i < m_executionOrder.size(); ++i)
            execIndex[m_executionOrder[i]] = i;

        for (const auto& chain : m_resourceChains)
        {
            if (chain.accesses.size() < 2)
                continue;

            auto sorted = chain.accesses;

            std::stable_sort(sorted.begin(), sorted.end(),
                [&](const ResourceAccess& a, const ResourceAccess& b)
                {
                    const uint32_t aIndex = execIndex[a.node];
                    const uint32_t bIndex = execIndex[b.node];

                    if (aIndex != bIndex)
                        return aIndex < bIndex;

                    if (a.firstUse != b.firstUse)
                        return a.firstUse;

                    return false;
                });

            ResourceAccess prev = sorted[0];

            for (size_t i = 1; i < sorted.size(); ++i)
            {
                const auto& curr = sorted[i];

                const bool needsBarrier =
                    prev.access == AccessType::Write ||
                    curr.access == AccessType::Write ||
                    prev.usage != curr.usage;

                if (needsBarrier)
                {
                    m_barriers.push_back({
                        .resource = chain.resource,
                        .fromNode = prev.node,
                        .toNode = curr.node,
                        .fromAccess = prev.access,
                        .toAccess = curr.access,
                        .oldUsage = prev.usage,
                        .newUsage = curr.usage
                        });
                }

                prev = curr;
            }
        }
    }

    void FrameGraphCompiler::buildNodeSchedules()
    {
        m_nodeSchedules.clear();
        m_incomingBarrierIndices.clear();
        m_outgoingBarrierIndices.clear();

        const size_t nodeCount = m_nodes.size();

        std::pmr::vector<uint32_t> incomingCounts(
            nodeCount,
            0,
            m_nodes.get_allocator());

        std::pmr::vector<uint32_t> outgoingCounts(
            nodeCount,
            0,
            m_nodes.get_allocator());

        for (const ResourceBarrier& barrier : m_barriers)
        {
            if (barrier.fromNode >= nodeCount ||
                barrier.toNode >= nodeCount)
            {
                continue;
            }

            ++outgoingCounts[barrier.fromNode];
            ++incomingCounts[barrier.toNode];
        }

        m_nodeSchedules.resize(nodeCount);

        uint32_t incomingOffset = 0;
        uint32_t outgoingOffset = 0;

        for (GraphNodeId node = 0; node < nodeCount; ++node)
        {
            NodeSchedule& schedule = m_nodeSchedules[node];
            schedule.node = node;

            schedule.firstIncomingBarrier = incomingOffset;
            schedule.incomingBarrierCount = incomingCounts[node];
            incomingOffset += incomingCounts[node];

            schedule.firstOutgoingBarrier = outgoingOffset;
            schedule.outgoingBarrierCount = outgoingCounts[node];
            outgoingOffset += outgoingCounts[node];
        }

        m_incomingBarrierIndices.resize(incomingOffset);
        m_outgoingBarrierIndices.resize(outgoingOffset);

        std::pmr::vector<uint32_t> incomingCursors(
            nodeCount,
            0,
            m_nodes.get_allocator());

        std::pmr::vector<uint32_t> outgoingCursors(
            nodeCount,
            0,
            m_nodes.get_allocator());

        for (uint32_t barrierIndex = 0;
             barrierIndex < m_barriers.size();
             ++barrierIndex)
        {
            const ResourceBarrier& barrier = m_barriers[barrierIndex];

            if (barrier.fromNode >= nodeCount ||
                barrier.toNode >= nodeCount)
            {
                continue;
            }

            NodeSchedule& fromSchedule =
                m_nodeSchedules[barrier.fromNode];
            NodeSchedule& toSchedule =
                m_nodeSchedules[barrier.toNode];

            const uint32_t outgoingIndex =
                fromSchedule.firstOutgoingBarrier +
                outgoingCursors[barrier.fromNode]++;
            m_outgoingBarrierIndices[outgoingIndex] = barrierIndex;

            const uint32_t incomingIndex =
                toSchedule.firstIncomingBarrier +
                incomingCursors[barrier.toNode]++;
            m_incomingBarrierIndices[incomingIndex] = barrierIndex;
        }
    }

    static const char* toString(ResourceUsage usage)
    {
        switch (usage)
        {
        case ResourceUsage::ColorAttachment: return "ColorAttachment";
        case ResourceUsage::DepthAttachment: return "DepthAttachment";
        case ResourceUsage::SampledTexture:  return "SampledTexture";
        case ResourceUsage::ConstantBuffer: return "ConstantBuffer";
        case ResourceUsage::StorageBuffer: return "StorageBuffer";
        case ResourceUsage::StorageTexture: return "StorageTexture";
        case ResourceUsage::TransferDst: return "TransferDst";
        case ResourceUsage::TransferSrc: return "TransferSrc";
        case ResourceUsage::Present: return "Present";
            

        default: return "Unknown";
        }
    }

    void FrameGraphCompiler::debugLog()
    {
        spdlog::info("=== FrameGraph Compile ===");


        spdlog::info("Nodes:");
        for (const auto& node : m_nodes)
        {
            spdlog::info(
                "Node {} | Queue {} | Type {}",
                node.nodeId,
                static_cast<int>(node.queue),
                static_cast<int>(node.type));
        }

        spdlog::info("Resource Access Chains:");

        for (const auto& chain : m_resourceChains)
        {
            spdlog::info("Resource {}", chain.resource);

            for (const auto& access : chain.accesses)
            {
                spdlog::info(
                    "    Node {} | {}",
                    access.node,
                    access.access == AccessType::Write ? "WRITE" : "READ");
            }
        }
        for (const auto& chain : m_resourceChains)
        {
            for (size_t i = 1; i < chain.accesses.size(); ++i)
            {
                const auto& a = chain.accesses[i - 1];
                const auto& b = chain.accesses[i];

                if (a.node == b.node &&
                    !a.firstUse &&
                    !b.firstUse)
                {
                    spdlog::warn(
                        "Duplicate access detected on resource {} in node {}",
                        chain.resource,
                        a.node
                    );
                }
            }
        }

        spdlog::info("Dependencies:");
        for (const auto& dep : m_dependencies)
        {
            spdlog::info("{} -> {}", dep.source, dep.destination);
        }

        spdlog::info("Adjacency Lists:");

        for (GraphNodeId i = 0; i < m_nodes.size(); ++i)
        {
            spdlog::info("Node {}:", i);

            const uint32_t firstEdge = m_adjacencyOffsets[i];
            const uint32_t edgeCount = m_adjacencyCounts[i];

            if (edgeCount == 0)
            {
                spdlog::info("    (empty)");
                continue;
            }

            for (uint32_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex)
            {
                const GraphNodeId dst =
                    m_adjacencyEdges[firstEdge + edgeIndex];
                spdlog::info("    -> {}", dst);
            }
        }

        spdlog::info("Execution Order:");
        for (GraphNodeId id : m_executionOrder)
        {
            spdlog::info("{}", id);
        }

        spdlog::info("Execution Levels:");
        for (uint32_t levelIndex = 0; levelIndex < m_executionLevels.size(); ++levelIndex)
        {
            const ExecutionLevel& level = m_executionLevels[levelIndex];
            std::string nodes;
            for (uint32_t i = 0; i < level.nodeCount; ++i)
            {
                const GraphNodeId id =
                    m_executionLevelNodes[level.firstNode + i];
                nodes += std::to_string(id);
                nodes += " ";
            }

            spdlog::info("Level {} | {}", levelIndex, nodes);
        }

        spdlog::info("Resource Lifetimes:");
        for (const auto& lifetime : m_resourceLifetimes)
        {
            spdlog::info(
                "Resource {} | First {} | Last {}",
                lifetime.resource,
                lifetime.firstUse,
                lifetime.lastUse);
        }

        spdlog::info("Resource Barriers:");

        for (const auto& b : m_barriers)
        {
            spdlog::info(
                "Resource {} | "
                "Node {} [{}:{}] -> "
                "Node {} [{}:{}]",
                b.resource,
                b.fromNode,
                b.fromAccess == AccessType::Write ? "WRITE" : "READ",
                toString(b.oldUsage),
                b.toNode,
                b.toAccess == AccessType::Write ? "WRITE" : "READ",
                toString(b.newUsage)
            );
        }

        spdlog::info("Node Schedules:");

        for (const auto& schedule : m_nodeSchedules)
        {
            spdlog::info("Node {}:", schedule.node);

            // Incoming
            spdlog::info("Incoming Barriers:");

            if (schedule.incomingBarrierCount == 0)
            {
                spdlog::info("(none)");
            }
            else
            {
                for (uint32_t i = 0;
                     i < schedule.incomingBarrierCount;
                     ++i)
                {
                    const uint32_t index =
                        m_incomingBarrierIndices[
                            schedule.firstIncomingBarrier + i];
                    const auto& barrier = m_barriers[index];

                    spdlog::info(
                        "        Resource {} | {} -> {} | {} -> {}",
                        barrier.resource,
                        barrier.fromNode,
                        barrier.toNode,
                        toString(barrier.oldUsage),
                        toString(barrier.newUsage));
                }
            }

            // Outgoing
            spdlog::info("Outgoing Barriers:");

            if (schedule.outgoingBarrierCount == 0)
            {
                spdlog::info("(none)");
            }
            else
            {
                for (uint32_t i = 0;
                     i < schedule.outgoingBarrierCount;
                     ++i)
                {
                    const uint32_t index =
                        m_outgoingBarrierIndices[
                            schedule.firstOutgoingBarrier + i];
                    const auto& barrier = m_barriers[index];

                    spdlog::info(
                        " Resource {} | {} -> {} | {} -> {}",
                        barrier.resource,
                        barrier.fromNode,
                        barrier.toNode,
                        toString(barrier.oldUsage),
                        toString(barrier.newUsage));
                }
            }
        }

        // Validation 
        {
            std::vector<uint32_t> pos(m_nodes.size());

            for (uint32_t i = 0; i < m_executionOrder.size(); ++i)
            {
                pos[m_executionOrder[i]] = i;
            }

            for (const auto& dep : m_dependencies)
            {
                if (pos[dep.source] > pos[dep.destination])
                {
                    spdlog::error(
                        "INVALID ORDER: {} -> {} violated",
                        dep.source,
                        dep.destination);

                    assert(false);
                }
            }

            spdlog::info("Execution order validation PASSED");
        }

        spdlog::info("=========================");
    }
}
