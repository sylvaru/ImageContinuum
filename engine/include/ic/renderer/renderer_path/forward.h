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
        GraphResourceId environmentCubemap;

        void buildFrameGraph(FrameGraphBuilder& builder) override
        {
            // Resources
            backBuffer = builder.importTexture({
                .type = ImportedResource::Swapchain,
                .initialUsage = ResourceUsage::Present
                });
            sceneDepth = builder.createTexture();
            visibilityBuffer = builder.createBuffer();
            environmentCubemap = builder.createTexture({
                .width = 512,
                .height = 512,
                .depth = 1,
                .mipLevels = 1,
                .arrayLayers = 6,
                .cubeCompatible = true,
                .format = TextureFormat::RGBA32_Float,
                .usage =
                    TextureUsageFlags::Sampled |
                    TextureUsageFlags::Storage,
                .debugName = "Environment.SkyboxCubemap"
                });

            builder.addComputePass("Visibility")
                .pipeline("visibility_test")
                .dispatch(64)
                .write(visibilityBuffer, ResourceUsage::StorageBuffer);

            builder.addComputePass("IndependentCompute")
                .pipeline("independent_compute_test")
                .dispatch(64)
                .write(visibilityBuffer, ResourceUsage::StorageBuffer);

            EnvironmentConvertPassData environmentConvert{};
            environmentConvert.outputCubemap = environmentCubemap;
            auto environmentNode =
                builder.addGraphNode(
                    environmentConvert,
                    GraphNodeType::Compute,
                    QueueType::Compute);
            builder.write(
                environmentNode,
                environmentCubemap,
                ResourceUsage::StorageTexture);

            builder.addGraphicsPass("DepthPrepass")
                .pipeline("depth_prepass")
                .drawList(DrawListKind::SceneGeometry)
                .depth(sceneDepth);

            auto forwardNode = builder.addGraphicsPass("ForwardOpaque")
                .pipeline("forward_bindless")
                .drawList(DrawListKind::SceneGeometry)
                .depthLoadOp(AttachmentLoadOp::Load)
                .color(backBuffer)
                .depth(sceneDepth);

            builder.read(
                forwardNode,
                visibilityBuffer,
                ResourceUsage::StorageBuffer);

            auto skyboxNode = builder.addGraphicsPass("Skybox")
                .pipeline("skybox")
                .drawList(DrawListKind::Skybox)
                .colorLoadOp(AttachmentLoadOp::Load)
                .depthLoadOp(AttachmentLoadOp::Load)
                .color(backBuffer)
                .depth(sceneDepth);
            builder.read(
                skyboxNode,
                environmentCubemap,
                ResourceUsage::SampledTexture);

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
