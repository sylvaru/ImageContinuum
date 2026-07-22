// ic/renderer/frame_graph/frame_graph_types.h
#pragma once

#include "ic/renderer/render_types.h"

namespace ic
{

    using GraphNodeId = uint32_t;
    using GraphResourceId = uint32_t;

    inline constexpr GraphNodeId InvalidGraphNodeId = UINT32_MAX;
    inline constexpr GraphResourceId InvalidGraphResourceId = UINT32_MAX;

    enum class QueueType : uint8_t
    {
        Graphics,
        Compute,
        Transfer
    };

    enum class GraphNodeType : uint8_t
    {
        Graphics,
        Compute,
        Transfer,
        Present
    };

    enum class GraphResourceType : uint8_t
    {
        Texture,
        Buffer
    };

    enum class AccessType : uint8_t
    {
        Read,
        Write,
        ReadWrite
    };

    enum class ResourceUsage : uint8_t
    {
        SampledTexture,
        StorageTexture,

        ColorAttachment,
        DepthAttachment,

        VertexBuffer,
        IndexBuffer,

        ConstantBuffer,
        StorageBuffer,
        IndirectArgument,

        TransferSrc,
        TransferDst,

        Present
    };

    enum class ResourceOwnership : uint8_t
    {
        Transient,
        Persistent,
        Imported
    };

    // How many physical instances a graph resource is backed by across frames
    // in flight. This is orthogonal to ownership (transient vs persistent):
    //   - Single       : one physical resource reused every frame. Correct only
    //                     for resources whose cross-frame reuse is otherwise
    //                     serialized (persistent buffers, borrowed imports).
    //   - PerFrameSlot  : one physical resource per frame-in-flight slot. The
    //                     registry hands out the current slot's instance, so two
    //                     frames in flight never touch the same physical memory.
    //                     This is the default for transient resources.
    //   - History       : per-frame-slot instances PLUS explicit access to the
    //                     previous frame's instance (registry previousEntry),
    //                     for temporal / ping-pong techniques.
    enum class ResourceMultiplicity : uint8_t
    {
        Single,
        PerFrameSlot,
        History
    };

    // Physical instance selection within a resource's N-instance ring. Both are
    // driven by a monotonic per-submitted-frame counter (NOT frameIndex), so
    // skipped frames, where render() returns before recording, never advance
    // the ring and therefore never make the previous instance stale. current and
    // previous are always distinct when instanceCount > 1, and
    // previousFrameInstanceIndex(counter) == currentFrameInstanceIndex(counter-1)
    // so "previous" always resolves to the immediately preceding submitted frame.
    [[nodiscard]] inline uint32_t currentFrameInstanceIndex(
        uint64_t frameCounter, uint32_t instanceCount) noexcept
    {
        return instanceCount <= 1u
            ? 0u
            : static_cast<uint32_t>(frameCounter % instanceCount);
    }

    [[nodiscard]] inline uint32_t previousFrameInstanceIndex(
        uint64_t frameCounter, uint32_t instanceCount) noexcept
    {
        return instanceCount <= 1u
            ? 0u
            : static_cast<uint32_t>(
                  (frameCounter + instanceCount - 1u) % instanceCount);
    }

    enum class GraphResourceSemantic : uint8_t
    {
        Generic = 0,
        ClusterBounds,
        ClusterLightGrid,
        ClusterLightIndices,
        ClusterLightCounter,
        VisibleLights,
        PathTraceTonemap,
        // GPU-driven buffers owned by the graph registry and bound by both
        // backends. Inputs (InstanceBounds, DrawInputs) are CPU-uploaded per
        // frame slot; the rest are compute-cull outputs. IndirectArguments
        // carries a backend-defined native command stride (see
        // BufferDesc::elementCount).
        GpuDrivenInstanceBounds,
        GpuDrivenDrawInputs,
        GpuDrivenVisibleInstances,
        GpuDrivenVisibleCount,
        GpuDrivenIndirectArguments,
        GpuDrivenDrawMetadata,
        GpuDrivenBinCounts,
        GpuDrivenCullClassification,
        GpuDrivenCullStats,
        GpuDrivenCullStatsReadback,
        GiSurfaceData,
        GiSurfels,
        GiVisibilityMoments,
        GiHashBuckets,
        GiProbes,
        GiProbeStaging,
        GiAllocationQueue,
        GiPriorityQueue,
        GiIndirectArguments,
        GiCounters,
        GiDiagnostics,
        GiDiagnosticsReadback,
        GiResidualInterface,
        GiHitRecords,
        GiRawIrradiance,
        GiResolvedIrradiance,
        GiSceneDepth,
        GiSurfaceAttributes,
        RayTracingSceneToken
    };

    enum class ImportedResource : uint8_t
    {
        None,
        Swapchain
    };

    struct ResourceBarrier
    {
        GraphResourceId resource;

        GraphNodeId fromNode;
        GraphNodeId toNode;

        AccessType fromAccess;
        AccessType toAccess;

        ResourceUsage oldUsage;
        ResourceUsage newUsage;
        bool firstUse = false;
        // Targets the previous-frame instance of a history resource rather than
        // the current instance, so the backend transitions previousEntry().
        bool previousVersion = false;
    };

    struct ResourceAccess
    {
        GraphNodeId node;
        GraphResourceId resource;
        AccessType access;
        ResourceUsage usage;

        bool external = false;
        bool firstUse = false;
        // A read of the previous-frame version of a history resource. Such
        // accesses form their own resource chain in the compiler so they never
        // create an intra-frame dependency on this frame's writer of the same
        // resource (which would be a false dependency or a cycle).
        bool previousVersion = false;
    };

    struct ResourceState
    {
        ResourceUsage usage;
        AccessType access;
    };

    struct ImportedResourceDesc
    {
        ImportedResource type;
        ResourceUsage initialUsage;
    };

    struct GraphResource
    {
        GraphResourceId id;

        GraphResourceType type;
        ResourceOwnership ownership;
        ImportedResource imported;
        ResourceMultiplicity multiplicity = ResourceMultiplicity::Single;
        GraphResourceSemantic semantic = GraphResourceSemantic::Generic;
        ResourceUsage initialUsage;
        AccessType initialAccess;
        TextureDesc textureDesc;
        BufferDesc bufferDesc;

        uint32_t firstAccess;
        uint32_t accessCount;
    };

    struct ResourceLifetime
    {
        GraphResourceId resource;

        uint32_t firstUse;

        uint32_t lastUse;
    };

    struct Dependency
    {
        GraphNodeId source;
        GraphNodeId destination;
    };

    struct GraphNode
    {
        GraphNodeId id;

        QueueType queue;
        GraphNodeType type;

        // Author-declared: this pass is SAFE on the async compute queue. The
        // compiler copies it to ExecutionNode::asyncEligible, where the
        // scheduler decides whether to use it. See that field for why safety
        // and profitability are kept separate.
        bool asyncEligible = false;

        uint32_t payloadIndex;
    };

    struct NodeRecord
    {
        GraphNode graphNode;
    };

    struct NodeSchedule
    {
        GraphNodeId node;

        uint32_t firstIncomingBarrier = 0;
        uint32_t incomingBarrierCount = 0;

        uint32_t firstOutgoingBarrier = 0;
        uint32_t outgoingBarrierCount = 0;
    };

    struct ExecutionLevel
    {
        uint32_t firstNode = 0;
        uint32_t nodeCount = 0;
    };

    struct QueueSubmissionWait
    {
        uint32_t submissionIndex = 0;
    };

    struct QueueSubmissionBatch
    {
        QueueType queue = QueueType::Graphics;
        uint32_t levelIndex = 0;
        uint32_t firstNode = 0;
        uint32_t nodeCount = 0;
        uint32_t firstWait = 0;
        uint32_t waitCount = 0;
    };

    struct ExecutionNode
    {
        GraphNodeId nodeId;

        QueueType queue;

        GraphNodeType type;

        uint32_t firstBarrier;
        uint32_t barrierCount;

        uint32_t firstDependency;
        uint32_t dependencyCount;

        uint32_t payloadIndex;

        // Contiguous range in CompiledGraphPlan::resourceAccesses. Backends
        // consume these declared bindings directly; barriers are synchronization
        // output and must never be used to rediscover pass inputs/outputs.
        uint32_t firstResourceAccess = 0;
        uint32_t resourceAccessCount = 0;

        // Declares that this pass MAY run on the async compute queue: the path
        // has established that it carries no dependency on concurrent graphics
        // work and that its resources are safe to touch across a queue
        // boundary. It is a statement of correctness, not of profitability.
        //
        // Whether an eligible pass actually lands on the compute queue is the
        // scheduler's deterministic async-compute setting, because support can
        // only be decided from measurements the pass itself cannot see: async
        // costs an extra submission, fence signal and cross-queue wait per
        // batch, and it repays that only when the frame is GPU-bound and the
        // work genuinely overlaps. Eligible passes stay on graphics by default.
        bool asyncEligible = false;
    };

    // A cross-frame GPU ordering edge derived from a genuine shared-resource
    // hazard. It replaces the old blanket "every level-0 batch waits for the
    // previous frame's whole graph" rule: only submissions that actually touch a
    // resource whose physical memory is reused across frames are ordered, and
    // only against the specific previous-frame submission that last used it.
    //
    // Semantics: this frame's `consumerSubmission` must wait for the PREVIOUS
    // frame's `producerSubmission` to complete, but only when `consumerNode`
    // actually executes this frame (a cadence-skipped writer creates no hazard,
    // so per-frame-slot-only and skipped-write frames overlap freely).
    //   - Single/persistent written resource (WAR/WAW): consumerNode candidates
    //     cover cadence-conditional writers through the first per-frame writer;
    //     producerSubmission = the submission that last accessed the resource.
    //     The first candidate that executes therefore waits for the old last read.
    //   - History previous-instance read (RAW): consumerNode = the previous-
    //     version reader; producerSubmission = the submission that writes the
    //     current instance (the one the next frame reads).
    // Per-frame-slot and imported resources never appear here.
    struct CrossFrameDependency
    {
        GraphResourceId resource;
        GraphNodeId consumerNode;
        uint32_t consumerSubmission;
        uint32_t producerSubmission;
    };

    static_assert(sizeof(QueueType) == 1);
    static_assert(sizeof(GraphNodeType) == 1);
    static_assert(sizeof(GraphResourceType) == 1);
    static_assert(sizeof(AccessType) == 1);
    static_assert(sizeof(ResourceUsage) == 1);
    static_assert(sizeof(ResourceOwnership) == 1);
    static_assert(sizeof(ImportedResource) == 1);

    static_assert(sizeof(ResourceBarrier) == 20);
    static_assert(sizeof(ResourceAccess) == 16);
    static_assert(sizeof(ResourceState) == 2);
    static_assert(sizeof(ImportedResourceDesc) == 2);

    static_assert(sizeof(ResourceLifetime) == 12);
    static_assert(sizeof(Dependency) == 8);

    static_assert(sizeof(GraphNode) == 12);
    static_assert(sizeof(NodeRecord) == 12);
    static_assert(sizeof(NodeSchedule) == 20);

    static_assert(sizeof(ExecutionLevel) == 8);
    static_assert(sizeof(QueueSubmissionWait) == 4);
    static_assert(sizeof(QueueSubmissionBatch) == 24);
    static_assert(sizeof(ExecutionNode) == 40);
    static_assert(sizeof(CrossFrameDependency) == 16);
}
