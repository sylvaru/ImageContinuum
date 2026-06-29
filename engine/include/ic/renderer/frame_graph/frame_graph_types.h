// ic/renderer/frame_graph/frame_graph_types.h
#pragma once

namespace ic
{

    using GraphNodeId = uint32_t;
    using GraphResourceId = uint32_t;

    enum class QueueType : uint8_t
    {
        Graphics,
        Compute,
        Transfer
    };

    enum class GraphNodeType
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

        TransferSrc,
        TransferDst,

        Present
    };

    enum class ResourceOwnership : uint8_t
    {
        Transient,
        Imported
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
    };

    struct ResourceAccess
    {
        GraphNodeId node;
        GraphResourceId resource;
        AccessType access;
        ResourceUsage usage;

        bool external = false;
        bool firstUse = false;
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
        ResourceUsage initialUsage;
        AccessType initialAccess;

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

        uint32_t payloadIndex;
    };

    struct NodeRecord
    {
        GraphNode graphNode;
        std::pmr::vector<ResourceAccess> accesses;
    };

    struct NodeSchedule
    {
        GraphNodeId node;

        std::pmr::vector<uint32_t> incomingBarrierIndices;
        std::pmr::vector<uint32_t> outgoingBarrierIndices;
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
    };


}