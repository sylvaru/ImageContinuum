#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

#include "ic/renderer/render_types.h"

namespace ic
{
    struct PipelineId
    {
        uint32_t value = 0;

        explicit operator bool() const
        {
            return value != 0;
        }

        friend bool operator==(PipelineId lhs, PipelineId rhs)
        {
            return lhs.value == rhs.value;
        }
    };

    struct PipelineIdHash
    {
        size_t operator()(PipelineId id) const noexcept
        {
            return id.value;
        }
    };

    constexpr uint32_t pipelineIdValue(std::string_view name)
    {
        uint32_t hash = 2166136261u;
        for (const char c : name)
        {
            hash ^= static_cast<uint8_t>(c);
            hash *= 16777619u;
        }
        return hash == 0 ? 1u : hash;
    }

    constexpr PipelineId makePipelineId(std::string_view name)
    {
        return PipelineId{ pipelineIdValue(name) };
    }

    enum class VertexLayoutKind : uint8_t
    {
        Unknown = 0,
        AssetVertex
    };

    enum class PipelineBindingLayoutKind : uint8_t
    {
        Unknown = 0,
        Empty,
        ForwardBindless,
        ComputeStorageBuffer,
        EnvironmentConvert,
        Skybox,
        ClusteredForward,
        HiZDepthPyramid,
        GpuFrustumCull,
        PathTrace,
        PathTraceTonemap
    };

    enum class PrimitiveTopologyKind : uint8_t
    {
        TriangleList = 0
    };

    enum class CullMode : uint8_t
    {
        None = 0,
        Front,
        Back
    };

    enum class CompareOp : uint8_t
    {
        Never = 0,
        Less,
        Equal,
        LessEqual,
        Greater,
        Always
    };

    struct ShaderProgramDesc
    {
        std::filesystem::path vertexShader;
        std::filesystem::path pixelShader;
    };

    struct ComputeShaderProgramDesc
    {
        std::filesystem::path computeShader;
    };

    struct RasterStateDesc
    {
        CullMode cullMode = CullMode::Back;
        bool depthClip = true;
    };

    struct DepthStateDesc
    {
        bool enabled = false;
        bool write = false;
        CompareOp compare = CompareOp::LessEqual;
        TextureFormat format = TextureFormat::Unknown;
    };

    struct BlendStateDesc
    {
        bool enabled = false;
    };

    struct GraphicsPipelineDesc
    {
        std::string debugName;

        ShaderProgramDesc shaders;
        VertexLayoutKind vertexLayout = VertexLayoutKind::Unknown;
        PipelineBindingLayoutKind bindingLayout = PipelineBindingLayoutKind::Unknown;
        PrimitiveTopologyKind topology = PrimitiveTopologyKind::TriangleList;

        RasterStateDesc raster;
        DepthStateDesc depth;
        BlendStateDesc blend;

        std::array<TextureFormat, 8> colorFormats = {};
        uint32_t colorAttachmentCount = 0;
    };

    struct ComputePipelineDesc
    {
        std::string debugName;

        ComputeShaderProgramDesc shaders;
        PipelineBindingLayoutKind bindingLayout = PipelineBindingLayoutKind::Unknown;
    };

    struct GraphicsPipelineHandle
    {
        uint32_t index = UINT32_MAX;
        uint32_t generation = 0;

        explicit operator bool() const
        {
            return index != UINT32_MAX;
        }
    };

    struct ComputePipelineHandle
    {
        uint32_t index = UINT32_MAX;
        uint32_t generation = 0;

        explicit operator bool() const
        {
            return index != UINT32_MAX;
        }
    };
}
