// ic/rendering/render_types.h
#pragma once

#include <cstdint>

namespace ic 
{
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
        BGRA8_UNorm,
        BGRA8_SRGB,
        D32_Float
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
        ShaderDeviceAddress = 1u << 7
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
        const char* debugName = nullptr;
    };

    struct TextureDesc
    {
        uint32_t width = 1;
        uint32_t height = 1;
        uint32_t depth = 1;
        uint32_t mipLevels = 1;
        uint32_t arrayLayers = 1;
        TextureFormat format = TextureFormat::RGBA8_UNorm;
        TextureUsageFlags usage = TextureUsageFlags::Sampled;
        ResourceMemoryUsage memoryUsage = ResourceMemoryUsage::GpuOnly;
        const char* debugName = nullptr;
    };

    // A single, tightly packed indirect draw command (12 bytes)
    // Fed to GPU via DrawIndexedIndirect or ExecuteIndirect
    struct DrawCommand 
    {
        uint32_t pipelineIndex;
        uint32_t materialIndex; // Index into a global SSBO (Bindless)
        uint32_t meshIndex;     // Index into global vertex/index buffer (BDA)
    };

    // GPU Ring buffer
}
