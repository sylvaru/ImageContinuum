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
        m_resourceAccesses.clear();
        m_barriers.clear();
        m_resourceLifetimes.clear();
        m_dependencies.clear();
        m_executionOrder.clear();
        m_executionLevels.clear();
        m_executionLevelNodes.clear();
        m_queueSubmissions.clear();
        m_queueSubmissionNodes.clear();
        m_queueSubmissionWaits.clear();
        m_resourceChains.clear();
        m_nodeSchedules.clear();
        m_adjacencyOffsets.clear();
        m_adjacencyCounts.clear();
        m_adjacencyEdges.clear();
        m_incomingBarrierIndices.clear();
        m_outgoingBarrierIndices.clear();
  
        const auto nodes = builder.nodes();
        const auto accesses = builder.accesses();

        // Backends walk accesses as a contiguous range per node. Count and
        // scatter in two linear passes instead of rescanning every access for
        // every node as the graph grows.
        std::pmr::vector<uint32_t> accessCounts(
            nodes.size(), 0, m_nodes.get_allocator());
        for (const ResourceAccess& access : accesses)
        {
            assert(access.node < nodes.size());
            ++accessCounts[access.node];
        }

        m_nodes.reserve(nodes.size());
        m_resourceAccesses.resize(accesses.size());
        uint32_t firstAccess = 0;
        for (const auto& node : nodes)
        {
            ExecutionNode compiledNode{};

            compiledNode.nodeId = node.graphNode.id;
            compiledNode.queue = node.graphNode.queue;
            compiledNode.type = node.graphNode.type;
            compiledNode.payloadIndex = node.graphNode.payloadIndex;

            compiledNode.firstResourceAccess = firstAccess;
            compiledNode.resourceAccessCount = accessCounts[compiledNode.nodeId];
            firstAccess += compiledNode.resourceAccessCount;

            m_nodes.push_back(compiledNode);
        }

        std::pmr::vector<uint32_t> accessCursors(
            nodes.size(), 0, m_nodes.get_allocator());
        for (const ExecutionNode& node : m_nodes)
        {
            accessCursors[node.nodeId] = node.firstResourceAccess;
        }
        for (const ResourceAccess& access : accesses)
        {
            m_resourceAccesses[accessCursors[access.node]++] = access;
        }

        buildResourceAccessChains(builder);
        buildDependencies();
        buildAdjacencyLists();
        buildExecutionOrder();
        buildExecutionLevels();
        buildQueueSubmissions();
        buildCrossFrameDependencies(builder);
        buildResourceLifetimes();
        buildBarriers();
        buildNodeSchedules();

        if (spdlog::default_logger_raw() &&
            spdlog::default_logger_raw()->should_log(spdlog::level::trace))
        {
            debugLog();
        }

        auto resources = builder.resources();
        auto payloads = builder.payloads();

        return
        {
            .nodes = std::span<const ExecutionNode>(m_nodes),
            .executionOrder = std::span<const GraphNodeId>(m_executionOrder),
            .executionLevels = std::span<const ExecutionLevel>(m_executionLevels),
            .executionLevelNodes = std::span<const GraphNodeId>(m_executionLevelNodes),
            .queueSubmissions = std::span<const QueueSubmissionBatch>(m_queueSubmissions),
            .queueSubmissionNodes = std::span<const GraphNodeId>(m_queueSubmissionNodes),
            .queueSubmissionWaits = std::span<const QueueSubmissionWait>(m_queueSubmissionWaits),
            .dependencies = std::span<const Dependency>(m_dependencies),
            .barriers = std::span<const ResourceBarrier>(m_barriers),
            .nodeSchedules = std::span<const NodeSchedule>(m_nodeSchedules),
            .incomingBarrierIndices = std::span<const uint32_t>(m_incomingBarrierIndices),
            .outgoingBarrierIndices = std::span<const uint32_t>(m_outgoingBarrierIndices),
            .resourceLifetimes = std::span<const ResourceLifetime>(m_resourceLifetimes),
            .resourceAccesses = std::span<const ResourceAccess>(m_resourceAccesses),
            .resources = std::span<const GraphResource>(resources),
            .payloads = std::span<const PassPayload>(payloads),
            .crossFrameDependencies =
                std::span<const CrossFrameDependency>(m_crossFrameDependencies)
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

        for (const auto& chain : m_resourceChains)
        {
            LastWriter lastWriter{};
            std::pmr::vector<GraphNodeId> readers(
                m_nodes.get_allocator());

            for (const auto& access : chain.accesses)
            {
                if (access.access == AccessType::Write ||
                    access.access == AccessType::ReadWrite)
                {
                    if (lastWriter.valid &&
                        lastWriter.node != access.node)
                    {
                        m_dependencies.push_back({
                            .source = lastWriter.node,
                            .destination = access.node
                            });
                    }
                    for (GraphNodeId reader : readers)
                    {
                        if (reader != access.node)
                        {
                            m_dependencies.push_back({
                                .source = reader,
                                .destination = access.node
                                });
                        }
                    }
                    readers.clear();
                    lastWriter = { access.node, true };
                }
                else // Read
                {
                    if (lastWriter.valid &&
                        lastWriter.node != access.node)
                    {
                        m_dependencies.push_back({
                            .source = lastWriter.node,
                            .destination = access.node
                            });
                    }
                    if (!access.external &&
                        std::find(
                            readers.begin(), readers.end(), access.node) ==
                            readers.end())
                    {
                        readers.push_back(access.node);
                    }
                }
            }

            // A usage change is a state transition even when both accesses
            // are reads (for example sampled texture -> transfer source).
            // Keep the two nodes out of the same parallel recording level so
            // the barrier has a defined execution order and backend state
            // tracking is never mutated concurrently for that resource.
            for (size_t i = 1; i < chain.accesses.size(); ++i)
            {
                const ResourceAccess& previous = chain.accesses[i - 1];
                const ResourceAccess& current = chain.accesses[i];
                if (previous.external || current.external ||
                    previous.node == current.node ||
                    previous.usage == current.usage)
                {
                    continue;
                }

                m_dependencies.push_back({ previous.node, current.node });
            }

            // Exclusive resources require an ordered ownership handoff even
            // when both accesses are reads. This also orders read-only usage
            // changes such as sampled -> transfer source.
            for (size_t i = 1; i < chain.accesses.size(); ++i)
            {
                const ResourceAccess& previous = chain.accesses[i - 1];
                const ResourceAccess& current = chain.accesses[i];
                if (previous.external || current.external ||
                    previous.node == current.node ||
                    m_nodes[previous.node].queue == m_nodes[current.node].queue)
                {
                    continue;
                }

                m_dependencies.push_back({ previous.node, current.node });
            }
        }

        std::sort(
            m_dependencies.begin(), m_dependencies.end(),
            [](const Dependency& lhs, const Dependency& rhs)
            {
                return lhs.source < rhs.source ||
                    (lhs.source == rhs.source &&
                     lhs.destination < rhs.destination);
            });
        m_dependencies.erase(
            std::unique(
                m_dependencies.begin(), m_dependencies.end(),
                [](const Dependency& lhs, const Dependency& rhs)
                {
                    return lhs.source == rhs.source &&
                        lhs.destination == rhs.destination;
                }),
            m_dependencies.end());
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

    void FrameGraphCompiler::buildQueueSubmissions()
    {
        m_queueSubmissions.clear();
        m_queueSubmissionNodes.clear();
        m_queueSubmissionWaits.clear();

        m_nodeSubmission.assign(m_nodes.size(), UINT32_MAX);
        std::pmr::vector<uint32_t>& nodeSubmission = m_nodeSubmission;

        for (uint32_t levelIndex = 0;
             levelIndex < m_executionLevels.size();
             ++levelIndex)
        {
            const ExecutionLevel& level = m_executionLevels[levelIndex];
            for (QueueType queue : {
                     QueueType::Graphics,
                     QueueType::Compute,
                     QueueType::Transfer })
            {
                QueueSubmissionBatch batch{};
                batch.queue = queue;
                batch.levelIndex = levelIndex;
                batch.firstNode =
                    static_cast<uint32_t>(m_queueSubmissionNodes.size());

                for (uint32_t i = 0; i < level.nodeCount; ++i)
                {
                    const GraphNodeId node =
                        m_executionLevelNodes[level.firstNode + i];
                    if (m_nodes[node].queue == queue)
                    {
                        m_queueSubmissionNodes.push_back(node);
                    }
                }

                batch.nodeCount =
                    static_cast<uint32_t>(m_queueSubmissionNodes.size()) -
                    batch.firstNode;
                if (batch.nodeCount == 0)
                {
                    continue;
                }

                const uint32_t submissionIndex =
                    static_cast<uint32_t>(m_queueSubmissions.size());
                m_queueSubmissions.push_back(batch);
                for (uint32_t i = 0; i < batch.nodeCount; ++i)
                {
                    nodeSubmission[
                        m_queueSubmissionNodes[batch.firstNode + i]] =
                        submissionIndex;
                }
            }
        }

        // Store (destination, source) pairs in one flat allocation. A vector
        // per batch plus linear duplicate searches becomes quadratic for
        // graphs with many cross-queue edges.
        std::pmr::vector<uint64_t> waitEdges(m_nodes.get_allocator());
        waitEdges.reserve(m_dependencies.size());
        for (const Dependency& dependency : m_dependencies)
        {
            const uint32_t source = nodeSubmission[dependency.source];
            const uint32_t destination = nodeSubmission[dependency.destination];
            if (source == UINT32_MAX || destination == UINT32_MAX ||
                source == destination ||
                m_queueSubmissions[source].queue ==
                    m_queueSubmissions[destination].queue)
            {
                continue;
            }

            waitEdges.push_back(
                (static_cast<uint64_t>(destination) << 32u) | source);
        }

        std::sort(waitEdges.begin(), waitEdges.end());
        waitEdges.erase(
            std::unique(waitEdges.begin(), waitEdges.end()),
            waitEdges.end());

        size_t waitEdge = 0;
        for (uint32_t submissionIndex = 0;
             submissionIndex < m_queueSubmissions.size();
             ++submissionIndex)
        {
            QueueSubmissionBatch& batch =
                m_queueSubmissions[submissionIndex];
            batch.firstWait =
                static_cast<uint32_t>(m_queueSubmissionWaits.size());
            while (waitEdge < waitEdges.size() &&
                   static_cast<uint32_t>(waitEdges[waitEdge] >> 32u) ==
                       submissionIndex)
            {
                m_queueSubmissionWaits.push_back({
                    static_cast<uint32_t>(waitEdges[waitEdge]) });
                ++waitEdge;
            }
            batch.waitCount =
                static_cast<uint32_t>(m_queueSubmissionWaits.size()) -
                batch.firstWait;
        }
    }

    void FrameGraphCompiler::buildCrossFrameDependencies(
        const FrameGraphBuilder& builder)
    {
        m_crossFrameDependencies.clear();

        const auto resources = builder.resources();

        std::pmr::vector<uint32_t> execIndex(
            m_nodes.size(), 0, m_nodes.get_allocator());
        for (uint32_t i = 0; i < m_executionOrder.size(); ++i)
        {
            execIndex[m_executionOrder[i]] = i;
        }

        const auto submissionOf = [&](GraphNodeId node) -> uint32_t
        {
            return node < m_nodeSubmission.size()
                ? m_nodeSubmission[node]
                : UINT32_MAX;
        };

        // Scan a chain's REAL (non-synthetic) accesses for the earliest writer,
        // the latest access, and the latest writer, all in execution order.
        const auto scanChain =
            [&](const ResourceChain& chain,
                const ResourceAccess*& firstWriter,
                const ResourceAccess*& lastAccess,
                const ResourceAccess*& lastWriter)
        {
            firstWriter = nullptr;
            lastAccess = nullptr;
            lastWriter = nullptr;
            for (const ResourceAccess& access : chain.accesses)
            {
                if (access.external)
                {
                    // Synthetic "produced earlier" marker, not a real access.
                    continue;
                }
                const uint32_t idx = execIndex[access.node];
                if (!lastAccess || idx > execIndex[lastAccess->node])
                {
                    lastAccess = &access;
                }
                if (access.access != AccessType::Read)
                {
                    if (!firstWriter || idx < execIndex[firstWriter->node])
                    {
                        firstWriter = &access;
                    }
                    if (!lastWriter || idx > execIndex[lastWriter->node])
                    {
                        lastWriter = &access;
                    }
                }
            }
        };

        for (const ResourceChain& chain : m_resourceChains)
        {
            if (chain.accesses.empty() ||
                chain.resource >= resources.size())
            {
                continue;
            }
            const GraphResource& resource = resources[chain.resource];
            if (resource.ownership == ResourceOwnership::Imported)
            {
                // Imported (swapchain) cross-frame safety is owned by image
                // acquire + present, not the graph.
                continue;
            }

            const ResourceAccess* firstWriter = nullptr;
            const ResourceAccess* lastAccess = nullptr;
            const ResourceAccess* lastWriter = nullptr;
            scanChain(chain, firstWriter, lastAccess, lastWriter);

            if (chain.previousVersion)
            {
                // History RAW: this frame's previous-instance reader must wait
                // for the previous frame's current-instance writer. The current-
                // version chain lives at index == resource id.
                if (chain.resource >= m_resourceChains.size())
                {
                    continue;
                }
                const ResourceAccess* curFirstWriter = nullptr;
                const ResourceAccess* curLastAccess = nullptr;
                const ResourceAccess* curLastWriter = nullptr;
                scanChain(
                    m_resourceChains[chain.resource],
                    curFirstWriter, curLastAccess, curLastWriter);
                if (!lastAccess || !curLastWriter)
                {
                    continue;
                }
                // consumer = first real previous-version read.
                const ResourceAccess* consumer = nullptr;
                for (const ResourceAccess& access : chain.accesses)
                {
                    if (access.external)
                    {
                        continue;
                    }
                    if (!consumer ||
                        execIndex[access.node] < execIndex[consumer->node])
                    {
                        consumer = &access;
                    }
                }
                const uint32_t consumerSub = submissionOf(consumer->node);
                const uint32_t producerSub = submissionOf(curLastWriter->node);
                if (consumerSub == UINT32_MAX || producerSub == UINT32_MAX)
                {
                    continue;
                }
                m_crossFrameDependencies.push_back({
                    .resource = chain.resource,
                    .consumerNode = consumer->node,
                    .consumerSubmission = consumerSub,
                    .producerSubmission = producerSub });
                continue;
            }

            // Current-version chain. Per-frame-slot (and the current instance of
            // a history resource) get a fresh physical instance each frame, so
        // adjacent frames never collide. CPU frame-slot pacing already
            // guards the ring reuse. Only Single/persistent instances are shared.
            if (resource.multiplicity != ResourceMultiplicity::Single)
            {
                continue;
            }
            // Read-only-across-frames persistent resources (no writer) are safe.
            if (!firstWriter || !lastAccess)
            {
                continue;
            }
            const uint32_t consumerSub = submissionOf(firstWriter->node);
            const uint32_t producerSub = submissionOf(lastAccess->node);
            if (consumerSub == UINT32_MAX || producerSub == UINT32_MAX)
            {
                continue;
            }
            m_crossFrameDependencies.push_back({
                .resource = chain.resource,
                .consumerNode = firstWriter->node,
                .consumerSubmission = consumerSub,
                .producerSubmission = producerSub });
        }
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

            // The previous-version chain shares the resource's physical lifetime
            // with its current-version chain; it must not emit a second, shorter
            // lifetime for the same resource id.
            if (chain.previousVersion)
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

        const auto resources = builder.resources();
        const auto accesses = builder.accesses();

        // GraphResourceId is a dense builder-owned index, so direct indexing
        // avoids hash computation, per-entry allocations, and a second
        // unordered-map-to-vector flattening pass.
        //
        // Current-version accesses form one chain per resource (indexed by id).
        // Previous-version reads (history resources) form an additional chain
        // per resource, appended after the current chains, so the two frame
        // versions are analyzed as distinct instances.
        std::pmr::vector<uint32_t> currentCounts(
            resources.size(), 0, m_nodes.get_allocator());
        std::pmr::vector<uint32_t> previousCounts(
            resources.size(), 0, m_nodes.get_allocator());
        for (const ResourceAccess& access : accesses)
        {
            assert(access.resource < resources.size());
            if (access.previousVersion)
            {
                ++previousCounts[access.resource];
            }
            else
            {
                ++currentCounts[access.resource];
            }
        }

        m_resourceChains.reserve(resources.size() * 2u);
        for (GraphResourceId resource = 0;
             resource < resources.size();
             ++resource)
        {
            m_resourceChains.emplace_back(resource, builder.get_allocator());
            m_resourceChains.back().accesses.reserve(
                static_cast<size_t>(currentCounts[resource]) + 1u);
        }

        // Map each resource with previous-version reads to its appended chain.
        std::pmr::vector<uint32_t> previousChainIndex(
            resources.size(), UINT32_MAX, m_nodes.get_allocator());
        for (GraphResourceId resource = 0;
             resource < resources.size();
             ++resource)
        {
            if (previousCounts[resource] == 0)
            {
                continue;
            }
            previousChainIndex[resource] =
                static_cast<uint32_t>(m_resourceChains.size());
            m_resourceChains.emplace_back(resource, builder.get_allocator());
            m_resourceChains.back().previousVersion = true;
            m_resourceChains.back().accesses.reserve(
                static_cast<size_t>(previousCounts[resource]) + 1u);
        }

        for (const ResourceAccess& access : accesses)
        {
            const uint32_t chainIndex = access.previousVersion
                ? previousChainIndex[access.resource]
                : access.resource;
            ResourceChain& chain = m_resourceChains[chainIndex];
            if (chain.accesses.empty())
            {
                const GraphResource& resource = resources[access.resource];
                // The previous-version chain's synthetic first access models
                // "produced by the previous frame": external (creates no
                // intra-frame dependency) and firstUse. Its physical instance's
                // real prior state is carried by the registry, and the executor's
                // cross-frame ordering guarantees that producer has completed.
                chain.accesses.push_back({
                    .node = access.node,
                    .resource = access.resource,
                    .access = resource.ownership == ResourceOwnership::Imported
                        ? resource.initialAccess : AccessType::Read,
                    .usage = resource.ownership == ResourceOwnership::Imported
                        ? resource.initialUsage : access.usage,
                    .external = true,
                    .firstUse = true,
                    .previousVersion = access.previousVersion
                });
            }
            chain.accesses.push_back(access);
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

                bool needsBarrier =
                    prev.access != AccessType::Read ||
                    curr.access != AccessType::Read ||
                    prev.usage != curr.usage;

                // The previous-version chain's first real access must always
                // emit a barrier: the previous physical instance was last left
                // in whatever state it held as a current instance (e.g. a UAV
                // write or a different layout), so it needs an explicit
                // transition into the read state on its own instance. The
                // backend resolves the real "before" state from the registry.
                if (chain.previousVersion && i == 1)
                {
                    needsBarrier = true;
                }

                if (needsBarrier)
                {
                    m_barriers.push_back({
                        .resource = chain.resource,
                        .fromNode = prev.node,
                        .toNode = curr.node,
                        .fromAccess = prev.access,
                        .toAccess = curr.access,
                        .oldUsage = prev.usage,
                        .newUsage = curr.usage,
                        .firstUse = prev.firstUse,
                        .previousVersion = chain.previousVersion
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
        case ResourceUsage::IndirectArgument: return "IndirectArgument";
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
