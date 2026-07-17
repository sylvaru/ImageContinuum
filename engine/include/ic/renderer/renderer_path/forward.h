// ic/renderer/renderer_path/forward_path.h
#pragma once
#include "renderer_path.h"
#include "ic/renderer/frame_graph/frame_graph_types.h"
#include "ic/renderer/frame_graph/frame_graph_builder.h"
#include "ic/renderer/gpu_driven_submission.h"

namespace ic
{
    class ForwardRendererPath : public RendererPath
    {
    public:

        GraphResourceId backBuffer;
        GraphResourceId sceneDepth;
        GraphResourceId depthPyramid;
        GraphResourceId environmentCubemap;
        GpuDrivenGraphResources gpuDriven;

        void buildFrameGraph(
            RendererPathContext& ctx,
            FrameGraphBuilder& builder) override
        {
            uint32_t renderWidth = ctx.renderExtent.width;
            uint32_t renderHeight = ctx.renderExtent.height;
            const uint32_t hiZMipCount =
                1 + static_cast<uint32_t>(
                    std::floor(std::log2(
                        std::max(renderWidth, renderHeight))));

            // Resources
            backBuffer = builder.importTexture({
                .type = ImportedResource::Swapchain,
                .initialUsage = ResourceUsage::Present
                });
            sceneDepth = builder.createTexture({
                .width = renderWidth,
                .height = renderHeight,
                .depth = 1,
                .mipLevels = 1,
                .arrayLayers = 1,
                .format = TextureFormat::D32_Float,
                .usage =
                    TextureUsageFlags::DepthAttachment |
                    TextureUsageFlags::Sampled,
                    .debugName = "Forward.SceneDepth"
                });
            depthPyramid = builder.createHistoryTexture({
                .width = renderWidth,
                .height = renderHeight,
                .depth = 1,
                .mipLevels = hiZMipCount,
                .arrayLayers = 1,
                .format = TextureFormat::R32_Float,
                .usage =
                    TextureUsageFlags::Sampled |
                    TextureUsageFlags::Storage,
                .debugName = "Forward.DepthPyramid"
                });

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

            gpuDriven = declareGpuDrivenResources(
                builder, ctx.occlusionDiagnosticsEnabled);
            addGpuDrivenCullPasses(builder, gpuDriven, depthPyramid);

            auto depthNode = builder.addGraphicsPass("DepthPrepass")
                .pipeline("depth_prepass")
                .drawList(DrawListKind::SceneGeometry)
                .depth(sceneDepth);
            readGpuDrivenStream(builder, depthNode, gpuDriven);

            builder.addComputePass("BuildHiZ")
                .pipeline("build_hiz")
                .dispatch((renderWidth + 7) / 8, (renderHeight + 7) / 8, 1)
                .read(sceneDepth, ResourceUsage::SampledTexture)
                .write(depthPyramid, ResourceUsage::StorageTexture);
            addGpuDrivenOcclusionValidationPasses(
                builder,
                gpuDriven,
                depthPyramid,
                ctx.asyncComputeEnabled
                    ? QueueType::Compute
                    : QueueType::Graphics);

            auto forwardNode = builder.addGraphicsPass("ForwardOpaque")
                .pipeline("forward_bindless")
                .drawList(DrawListKind::SceneGeometry)
                .depthLoadOp(AttachmentLoadOp::Load)
                .color(backBuffer)
                .depth(sceneDepth);
            readGpuDrivenStream(builder, forwardNode, gpuDriven);

            //builder.read(
            //    forwardNode,
            //    visibilityBuffer,
            //    ResourceUsage::StorageBuffer);

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
