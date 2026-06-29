// ic/renderer/renderer_path/forward_path.h
#pragma once
#include "renderer_path.h"
#include "ic/renderer/frame_graph/frame_graph_types.h"
#include "ic/renderer/frame_graph/frame_graph_builder.h"

namespace ic
{
    class ForwardRendererPath : public RendererPath
    {
    public:

        GraphResourceId backBuffer;

        void buildFrameGraph(FrameGraphBuilder& builder) override
        {
            // Resources
            backBuffer = builder.importTexture({
                .type = ImportedResource::Swapchain,
                .initialUsage = ResourceUsage::Present
                });

            auto clearNode =
                builder.addGraphNode(
                    ClearPassData{},
                    GraphNodeType::Graphics,
                    QueueType::Graphics);

            builder.write(
                clearNode,
                backBuffer,
                ResourceUsage::ColorAttachment);

            auto presentNode =
                builder.addGraphNode(
                    PresentPassData{},
                    GraphNodeType::Present,
                    QueueType::Graphics);

            builder.read(
                presentNode,
                backBuffer,
                ResourceUsage::Present);
        }
    };

}