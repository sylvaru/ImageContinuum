// ic/rendering/render_types.h
#pragma once

#include <cstdint>

#include <glm/glm.hpp>

namespace ic 
{
    constexpr uint32_t MaxBindlessTextures = 4096;
    constexpr uint32_t MaxBindlessSamplers = 256;
    constexpr uint32_t ClusteredForwardTileSizeX = 16;
    constexpr uint32_t ClusteredForwardTileSizeY = 16;
    constexpr uint32_t ClusteredForwardSliceCountZ = 24;
    constexpr uint32_t ClusteredForwardMaxLightsPerCluster = 128;
    constexpr uint32_t ClusteredForwardMaxVisibleLights = 256;
    constexpr uint32_t ClusteredForwardMaxGpuCullInstances = 65536;

    enum class GpuCullDebugMode : uint32_t
    {
        Off = 0,
        Statistics,
        Classification
    };

    enum class GpuCullClassification : uint32_t
    {
        Visible = 0,
        FrustumCulled,
        OcclusionCulled,
        ConservativeRetained
    };

    struct GpuCullStats
    {
        uint32_t inputCount = 0;
        uint32_t frustumCulled = 0;
        uint32_t occlusionCulled = 0;
        uint32_t conservativeRetained = 0;

        uint32_t visible = 0;
        uint32_t overflow = 0;
        uint32_t falseOccluded = 0;
        uint32_t falseVisible = 0;

        uint32_t historyValid = 0;
        uint32_t occlusionEnabled = 0;
        uint32_t padding0 = 0;
        uint32_t padding1 = 0;
    };
    static_assert(sizeof(GpuCullStats) == 48);

    struct GpuCullPerformance
    {
        double gpuMilliseconds = 0.0;
        double cpuRecordMilliseconds = 0.0;
    };

    enum class ResourceMemoryUsage : uint8_t
    {
        GpuOnly,
        CpuToGpu,
        GpuToCpu
    };

    enum class TextureFormat : uint8_t
    {
        Unknown,
        RGBA8_UNorm,
        RGBA8_SRGB,
        RGBA32_Float,
        BGRA8_UNorm,
        BGRA8_SRGB,
        D32_Float,
        R32_Float

    };

    enum class BufferUsageFlags : uint32_t
    {
        None = 0,
        Vertex = 1u << 0,
        Index = 1u << 1,
        Constant = 1u << 2,
        Storage = 1u << 3,
        TransferSrc = 1u << 4,
        TransferDst = 1u << 5,
        Indirect = 1u << 6,
        ShaderDeviceAddress = 1u << 7,
        AccelerationStructureStorage = 1u << 8,
        AccelerationStructureBuildInput = 1u << 9,
        AccelerationStructureScratch = 1u << 10
    };

    enum class TextureUsageFlags : uint32_t
    {
        None = 0,
        Sampled = 1u << 0,
        Storage = 1u << 1,
        ColorAttachment = 1u << 2,
        DepthAttachment = 1u << 3,
        TransferSrc = 1u << 4,
        TransferDst = 1u << 5
    };

    struct alignas(16) GpuClusterBounds
    {
        // View-space minimum corner of the cluster AABB.
        // xyz = min bounds
        // w   = unused / padding
        glm::vec4 minBounds;

        // View-space maximum corner of the cluster AABB.
        // xyz = max bounds
        // w   = unused / padding
        glm::vec4 maxBounds;
    };

    static_assert(sizeof(GpuClusterBounds) == 32);
    static_assert(alignof(GpuClusterBounds) == 16);


    struct alignas(16) GpuClusterLightGrid
    {
        // Offset into clusterLightIndexBuffer.
        uint32_t offset = 0;

        // Number of light indices used by this cluster.
        uint32_t count = 0;

        // Keep the struct 16-byte aligned and shader-friendly.
        uint32_t pad0 = 0;
        uint32_t pad1 = 0;
    };

    static_assert(sizeof(GpuClusterLightGrid) == 16);
    static_assert(alignof(GpuClusterLightGrid) == 16);

    struct alignas(16) GpuVisibleLight
    {
        glm::vec4 positionRange = glm::vec4(0.0f);
        glm::vec4 colorIntensity = glm::vec4(0.0f);
    };

    static_assert(sizeof(GpuVisibleLight) == 32);
    static_assert(alignof(GpuVisibleLight) == 16);

    struct alignas(16) GpuInstanceBounds
    {
        // xyz = world-space sphere center, w = sphere radius.
        glm::vec4 centerRadius = glm::vec4(0.0f);
    };

    static_assert(sizeof(GpuInstanceBounds) == 16);
    static_assert(alignof(GpuInstanceBounds) == 16);



    constexpr BufferUsageFlags operator|(
        BufferUsageFlags lhs,
        BufferUsageFlags rhs)
    {
        return static_cast<BufferUsageFlags>(
            static_cast<uint32_t>(lhs) |
            static_cast<uint32_t>(rhs));
    }

    constexpr BufferUsageFlags operator&(
        BufferUsageFlags lhs,
        BufferUsageFlags rhs)
    {
        return static_cast<BufferUsageFlags>(
            static_cast<uint32_t>(lhs) &
            static_cast<uint32_t>(rhs));
    }

    constexpr TextureUsageFlags operator|(
        TextureUsageFlags lhs,
        TextureUsageFlags rhs)
    {
        return static_cast<TextureUsageFlags>(
            static_cast<uint32_t>(lhs) |
            static_cast<uint32_t>(rhs));
    }

    constexpr TextureUsageFlags operator&(
        TextureUsageFlags lhs,
        TextureUsageFlags rhs)
    {
        return static_cast<TextureUsageFlags>(
            static_cast<uint32_t>(lhs) &
            static_cast<uint32_t>(rhs));
    }

    constexpr bool hasFlag(
        BufferUsageFlags value,
        BufferUsageFlags flag)
    {
        return (static_cast<uint32_t>(value & flag)) != 0;
    }

    constexpr bool hasFlag(
        TextureUsageFlags value,
        TextureUsageFlags flag)
    {
        return (static_cast<uint32_t>(value & flag)) != 0;
    }

    struct BufferDesc
    {
        uint64_t size = 0;
        BufferUsageFlags usage = BufferUsageFlags::None;
        ResourceMemoryUsage memoryUsage = ResourceMemoryUsage::GpuOnly;
        bool mappedAtCreation = false;
        // Element count for buffers whose per-element byte stride is defined by
        // the backend rather than by an API-neutral struct (currently only the
        // GPU-driven indirect-command buffer: DX12 packs root constants into a
        // 36-byte command, Vulkan uses a 20-byte VkDrawIndexedIndirectCommand).
        // When non-zero and the resource carries a native-stride semantic, the
        // backend registry sizes the buffer as elementCount * nativeStride and
        // ignores `size`. Zero means `size` is authoritative.
        uint32_t elementCount = 0;
        const char* debugName = nullptr;
    };

    struct TextureDesc
    {
        uint32_t width = 1;
        uint32_t height = 1;
        uint32_t depth = 1;
        uint32_t mipLevels = 1;
        uint32_t arrayLayers = 1;
        bool cubeCompatible = false;
        TextureFormat format = TextureFormat::RGBA8_UNorm;
        TextureUsageFlags usage = TextureUsageFlags::Sampled;
        ResourceMemoryUsage memoryUsage = ResourceMemoryUsage::GpuOnly;
        const char* debugName = nullptr;
    };

    // Engine-side draw classification metadata. This is intentionally not an
    // API indirect argument structure.
    struct DrawCommand 
    {
        uint32_t pipelineIndex;
        uint32_t materialIndex; // Index into a global SSBO (Bindless)
        uint32_t meshIndex;     // Index into global vertex/index buffer (BDA)
    };

    // Common binary layout shared by VkDrawIndexedIndirectCommand and
    // D3D12_DRAW_INDEXED_ARGUMENTS. Keep API types out of renderer paths.
    struct GpuIndexedIndirectArguments
    {
        uint32_t indexCount = 0;
        uint32_t instanceCount = 0;
        uint32_t firstIndex = 0;
        int32_t vertexOffset = 0;
        uint32_t firstInstance = 0;
    };
    static_assert(sizeof(GpuIndexedIndirectArguments) == 20);

    struct GpuDrawMetadata
    {
        uint32_t meshIndex = 0;
        uint32_t materialIndex = 0;
        uint32_t transformIndex = 0;
        uint32_t instanceIndex = 0;

        uint32_t geometryRangeIndex = 0;
        uint32_t pipelineBinIndex = 0;
        uint32_t materialBinIndex = 0;
        uint32_t geometryBinIndex = 0;

        uint32_t cullState = 0;
        uint32_t padding0 = 0;
        uint32_t padding1 = 0;
        uint32_t padding2 = 0;
    };
    static_assert(sizeof(GpuDrawMetadata) == 48);

    struct GpuDrawInput
    {
        GpuDrawMetadata metadata = {};
        uint32_t indexCount = 0;
        uint32_t firstIndex = 0;
        int32_t vertexOffset = 0;
        uint32_t commandBinOffset = 0;
        uint32_t commandBinCapacity = 0;
        uint32_t commandBinLocalIndex = 0;
        uint32_t padding1 = 0;
        uint32_t padding2 = 0;
    };
    static_assert(sizeof(GpuDrawInput) == 80);

    // GPU Ring buffer
}
