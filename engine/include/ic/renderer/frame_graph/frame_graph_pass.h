// ic/renderer/frame_graph/graph_pass.h
#pragma once
#include <string>

#include "ic/renderer/render_pipeline.h"

namespace ic
{
    class FrameGraphBuilder;

    struct GraphicsPassData
    {
        std::string name;
        PipelineId pipeline = {};
    };

    struct TransferPassData
    {
        std::string name;
    };

    struct ComputePassData
    {
        std::string name;
        PipelineId pipeline = {};
        uint32_t groupCountX = 1;
        uint32_t groupCountY = 1;
        uint32_t groupCountZ = 1;
    };

    struct ClearPassData {};

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
        ComputePassData,
        TransferPassData,
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
