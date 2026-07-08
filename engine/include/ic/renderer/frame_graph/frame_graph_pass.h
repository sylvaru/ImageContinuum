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

    struct GraphicsPassData
    {
        std::string name;
        PipelineId pipeline = {};
        DrawListKind drawList = DrawListKind::None;
        AttachmentLoadOp colorLoadOp = AttachmentLoadOp::Clear;
        AttachmentLoadOp depthLoadOp = AttachmentLoadOp::Clear;
    };

    struct TransferPassData
    {
        std::string name;
        GraphResourceId source = InvalidGraphResourceId;
        GraphResourceId destination = InvalidGraphResourceId;
    };

    struct ComputePassData
    {
        std::string name;
        PipelineId pipeline = {};
        uint32_t groupCountX = 1;
        uint32_t groupCountY = 1;
        uint32_t groupCountZ = 1;
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
