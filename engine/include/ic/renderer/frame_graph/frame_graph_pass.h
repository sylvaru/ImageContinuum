// ic/renderer/frame_graph/graph_pass.h
#pragma once

namespace ic
{
    class FrameGraphBuilder;


    struct GraphicsPassData
    {
        std::string_view name;

        // Eventually:
        //bool depthTest;
        //bool depthWrite;
        //PipelineHandle pipeline;
    };

    struct ComputePassData
    {
        std::string_view name;
    };

    template<typename T>
    concept GraphPass = requires(
        T pass,
        FrameGraphBuilder & builder)
    {
        { pass.setup(builder) } -> std::same_as<void>;
    };
}