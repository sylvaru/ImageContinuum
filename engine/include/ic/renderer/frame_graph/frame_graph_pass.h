// ic/renderer/frame_graph/graph_pass.h
#pragma once
#include <concepts>
#include <cstdint>
#include <string>
#include <variant>

#include "ic/renderer/frame_graph/frame_graph_types.h"
#include "ic/renderer/render_pipeline.h"

namespace ic
{
    class FrameGraphBuilder;

    enum class DrawListKind : uint8_t
    {
        None = 0,
        SceneGeometry,
        Skybox,
        TransparentGeometry

    };

    enum class AttachmentLoadOp : uint8_t
    {
        Clear = 0,
        Load,
        DontCare
    };

    enum class PassInvalidation : uint32_t
    {
        None          = 0,
        Startup       = 1u << 0,
        GraphRebuild  = 1u << 1,
        Resize        = 1u << 2,
        Resources     = 1u << 3,
        Configuration = 1u << 4,
        Scene         = 1u << 5,
        Environment   = 1u << 6,
        Manual        = 1u << 7
    };

    constexpr PassInvalidation operator|(
        PassInvalidation lhs, PassInvalidation rhs) noexcept
    {
        return static_cast<PassInvalidation>(
            static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
    }

    constexpr PassInvalidation& operator|=(
        PassInvalidation& lhs, PassInvalidation rhs) noexcept
    {
        lhs = lhs | rhs;
        return lhs;
    }

    constexpr bool any(
        PassInvalidation value, PassInvalidation mask) noexcept
    {
        return (static_cast<uint32_t>(value) &
                static_cast<uint32_t>(mask)) != 0;
    }

    // Controls pass-body execution. Barriers and queue ordering remain in the
    // compiled graph even when a body is skipped, so persistent outputs retain
    // valid native state without rebuilding the graph.
    enum class PassCadence : uint8_t
    {
        PerFrame = 0,
        Once,
        OnInvalidation,
        OnResize // compatibility shorthand for Resize invalidation
    };

    struct PassExecutionPolicy
    {
        PassCadence cadence = PassCadence::PerFrame;
        PassInvalidation invalidation = PassInvalidation::None;
    };

    struct GraphicsPassData
    {
        std::string name;
        PipelineId pipeline = {};
        DrawListKind drawList = DrawListKind::None;
        AttachmentLoadOp colorLoadOp = AttachmentLoadOp::Clear;
        AttachmentLoadOp depthLoadOp = AttachmentLoadOp::Clear;
        PassExecutionPolicy execution = {};
    };

    struct TransferPassData
    {
        std::string name;
        GraphResourceId source = InvalidGraphResourceId;
        GraphResourceId destination = InvalidGraphResourceId;
        PassExecutionPolicy execution = {};
    };

    struct ComputePassData
    {
        std::string name;
        PipelineId pipeline = {};
        uint32_t groupCountX = 1;
        uint32_t groupCountY = 1;
        uint32_t groupCountZ = 1;
        PassExecutionPolicy execution = {};
    };

    struct EnvironmentConvertPassData
    {
        uint32_t cubemapSize = 512;
        uint32_t groupSizeX = 8;
        uint32_t groupSizeY = 8;
        uint32_t groupCountX = 64;
        uint32_t groupCountY = 64;
        uint32_t groupCountZ = 6;
        GraphResourceId outputCubemap = InvalidGraphResourceId;
        PipelineId pipeline = makePipelineId("equirect_to_cubemap");
    };

    struct ClearPassData {};

    struct PathTracePassData
    {
        uint32_t width = 0;
        uint32_t height = 0;

        uint32_t groupSizeX = 8;
        uint32_t groupSizeY = 8;
        uint32_t groupCountX = 1;
        uint32_t groupCountY = 1;
        uint32_t groupCountZ = 1;

        GraphResourceId outputAccumulation = InvalidGraphResourceId;
        GraphResourceId sceneBuffer = InvalidGraphResourceId;
        GraphResourceId instanceBuffer = InvalidGraphResourceId;
        GraphResourceId materialBuffer = InvalidGraphResourceId;
        GraphResourceId vertexBuffer = InvalidGraphResourceId;
        GraphResourceId indexBuffer = InvalidGraphResourceId;
        GraphResourceId bvhNodeBuffer = InvalidGraphResourceId;
        GraphResourceId bvhTriangleBuffer = InvalidGraphResourceId;

        PipelineId pipeline = makePipelineId("path_trace");
    };

    struct TonemapPassData
    {
        uint32_t width = 0;
        uint32_t height = 0;

        uint32_t groupSizeX = 8;
        uint32_t groupSizeY = 8;
        uint32_t groupCountX = 1;
        uint32_t groupCountY = 1;
        uint32_t groupCountZ = 1;

        GraphResourceId inputHDR = InvalidGraphResourceId;
        GraphResourceId outputBackBuffer = InvalidGraphResourceId;

        PipelineId pipeline = makePipelineId("path_trace_tonemap");
    };

    struct GeometryPassData
    {
        bool depthPrepass;
        bool occlusionCulling;
    };

    struct LightingPassData {};
    struct PresentPassData {};
    struct ShadowPassData {};
    struct PostProcessPassData {};

    using PassPayload =
        std::variant<
        ClearPassData,
        GraphicsPassData,
        GeometryPassData,
        LightingPassData,
        EnvironmentConvertPassData,
        ComputePassData,
        TransferPassData,
        PathTracePassData,
        TonemapPassData,
        PresentPassData,
        ShadowPassData,
        PostProcessPassData>;

    template<typename T>
    concept GraphPass = requires(
        T pass,
        FrameGraphBuilder & builder)
    {
        { pass.setup(builder) } -> std::same_as<void>;
    };

}



// Eventually PassData should have:
//bool depthTest;
//bool depthWrite;
//PipelineHandle pipeline;
