// ic/renderer/frame_graph/graph_pass.h
#pragma once

namespace ic
{
    class FrameGraphBuilder;

    struct GraphicsPassData {};

    struct ComputePassData {};

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
        GeometryPassData,
        LightingPassData,
        ComputePassData,
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