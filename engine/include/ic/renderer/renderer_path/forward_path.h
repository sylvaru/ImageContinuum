#pragma once
#include "renderer_path.h"
#include "ic/renderer/frame_graph/frame_graph_types.h"
#include "ic/renderer/frame_graph/frame_graph_builder.h"

namespace ic
{
    class ForwardRendererPath : public RendererPath
    {
    public:
        GraphNodeId clearNode;
        GraphResourceId backBuffer;

        void buildFrameGraph(
            FrameGraphBuilder& builder) override
        {
            backBuffer = builder.createTexture();

            clearNode = builder.addGraphNode(
                GraphNodeType::Graphics,
                QueueType::Graphics);
            
            builder.write(
                clearNode,
                backBuffer,
                ResourceUsage::ColorAttachment);
        }
    };


}