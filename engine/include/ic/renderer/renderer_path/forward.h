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
        GraphResourceId sceneDepth;
        GraphResourceId visibilityBuffer;

        void buildFrameGraph(FrameGraphBuilder& builder) override
        {
            // Resources
            backBuffer = builder.importTexture({
                .type = ImportedResource::Swapchain,
                .initialUsage = ResourceUsage::Present
                });
            sceneDepth = builder.createTexture();
            visibilityBuffer = builder.createBuffer();

            builder.addComputePass("Visibility")
                .pipeline("visibility_test")
                .dispatch(64)
                .write(visibilityBuffer, ResourceUsage::StorageBuffer);

            builder.addComputePass("IndependentCompute")
                .pipeline("independent_compute_test")
                .dispatch(64)
                .write(visibilityBuffer, ResourceUsage::StorageBuffer);

            builder.addGraphicsPass("DepthPrepass")
                .pipeline("depth_prepass")
                .depth(sceneDepth);

            auto forwardNode = builder.addGraphicsPass("ForwardOpaque")
                .pipeline("forward_bindless")
                .color(backBuffer)
                .depth(sceneDepth);

            builder.read(
                forwardNode,
                visibilityBuffer,
                ResourceUsage::StorageBuffer);

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
