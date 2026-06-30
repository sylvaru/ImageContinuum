#pragma once
#include "renderer_path.h"
#include "ic/renderer/frame_graph/frame_graph_types.h"
#include "ic/renderer/frame_graph/frame_graph_builder.h"

namespace ic
{
    class PathTracerRendererPath : public RendererPath
    {
    public:

        GraphResourceId backBuffer;
        GraphResourceId sceneDepth;
        GraphResourceId visibilityBuffer;

        void buildFrameGraph([[maybe_unused]] FrameGraphBuilder& builder) override
        {
        
        }
    };

}
