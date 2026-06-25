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
        m_chainMap.clear();
        m_resourceChains.clear();
  

        auto nodes = builder.nodes();


        // Build node list
        for (const auto& node : nodes)
        {
            ExecutionNode compiledNode{};

            compiledNode.nodeId = node.graphNode.id;
            compiledNode.queue = node.graphNode.queue;
            compiledNode.type = node.graphNode.type;

            m_nodes.push_back(compiledNode);
        }

        buildResourceAccessChains(builder);
        buildDependencies();
        buildAdjacencyLists();
        buildExecutionOrder();
        buildResourceLifetimes();
        buildBarriers();

        debugLog();

        return {
            .nodes = std::span<const ExecutionNode>(m_nodes),
            .executionOrder = std::span<const GraphNodeId>(m_executionOrder),
            .dependencies = std::span<const Dependency>(m_dependencies),
            .barriers = std::span<const ResourceBarrier>(m_barriers),
            .resourceLifetimes = std::span<const ResourceLifetime>(m_resourceLifetimes)
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
        m_adjacencyLists.clear();
        m_adjacencyLists.resize(
            m_nodes.size(), 
            std::pmr::vector<GraphNodeId>(m_nodes.get_allocator()));

        for (const auto& dep : m_dependencies)
        {
            m_adjacencyLists[dep.source].push_back(dep.destination);
        }

        // Sort adjacency lists
        for (auto& list : m_adjacencyLists)
        {
            std::sort(list.begin(), list.end());
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

            for (GraphNodeId m : m_adjacencyLists[n])
            {
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

        for (const auto& node : builder.nodes())
        {
            for (const auto& access : node.accesses)
            {
                auto& chain =
                    m_chainMap.try_emplace(
                        access.resource,
                        access.resource,
                        builder.get_allocator()
                    ).first->second;

                chain.accesses.push_back(access);
            }
        }

        m_resourceChains.clear();
        m_resourceChains.reserve(m_chainMap.size());

        for (auto& [_, chain] : m_chainMap)
        {
            m_resourceChains.push_back(std::move(chain));
        }
    }

    void FrameGraphCompiler::buildBarriers()
    {
        m_barriers.clear();

        // Map node -> execution index
        std::pmr::vector<uint32_t> execIndex(
            m_nodes.size(),
            0,
            m_nodes.get_allocator()
        );

        for (uint32_t i = 0; i < m_executionOrder.size(); ++i)
        {
            execIndex[m_executionOrder[i]] = i;
        }

        // For each resource chain
        for (const auto& chain : m_resourceChains)
        {
            if (chain.accesses.size() < 2)
                continue;

            // Sort accesses by execution order
            auto sorted = chain.accesses;

            std::sort(sorted.begin(), sorted.end(),
                [&](const ResourceAccess& a, const ResourceAccess& b)
                {
                    return execIndex[a.node] < execIndex[b.node];
                });

            // Generate transitions
            for (size_t i = 1; i < sorted.size(); ++i)
            {
                const auto& prev = sorted[i - 1];
                const auto& curr = sorted[i];

                if (prev.access == curr.access &&
                    prev.usage == curr.usage)
                {
                    continue;
                }

                m_barriers.push_back({
                    .resource = chain.resource,
                    .fromNode = prev.node,
                    .toNode = curr.node,
                    .oldUsage = prev.usage,
                    .newUsage = curr.usage
                    });
            }
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

                if (a.node == b.node)
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
            const auto& list = m_adjacencyLists[i];

            spdlog::info("Node {}:", i);

            if (list.empty())
            {
                spdlog::info("    (empty)");
                continue;
            }

            for (GraphNodeId dst : list)
            {
                spdlog::info("    -> {}", dst);
            }
        }

        spdlog::info("Execution Order:");
        for (GraphNodeId id : m_executionOrder)
        {
            spdlog::info("{}", id);
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
                "Resource {} | {}:{} -> {}:{}",
                b.resource,
                b.fromNode,
                static_cast<int>(b.oldUsage),
                b.toNode,
                static_cast<int>(b.newUsage)
            );
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