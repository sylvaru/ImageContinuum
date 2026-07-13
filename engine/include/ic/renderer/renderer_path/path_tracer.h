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
        GraphResourceId environmentCubemap;
        GraphResourceId accumulationTexture;
        GraphResourceId tonemapTexture;
        GraphNodeId pathTraceNode = InvalidGraphNodeId;
        GraphNodeId tonemapNode = InvalidGraphNodeId;
        GraphNodeId copyNode = InvalidGraphNodeId;
        GraphNodeId presentNode = InvalidGraphNodeId;

        void buildFrameGraph(
            [[maybe_unused]] RendererPathContext& ctx,
            FrameGraphBuilder& builder) override
        {
            backBuffer = builder.importTexture({
                .type = ImportedResource::Swapchain,
                .initialUsage = ResourceUsage::Present
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
                .debugName = "Environment.PathTracerCubemap"
                });

            // Environment conversion is submitted only when the source or bake
            // settings change by Renderer::IBLBaker, outside the recurrent graph.

            accumulationTexture = builder.createTexture({
                .width = 1,
                .height = 1,
                .format = TextureFormat::RGBA32_Float,
                .usage =
                    TextureUsageFlags::Storage |
                    TextureUsageFlags::Sampled,
                .debugName = "PathTracer.AccumulationHDR"
                });

            tonemapTexture = builder.createTexture({
                .width = 1,
                .height = 1,
                .format = TextureFormat::RGBA8_UNorm,
                .usage =
                    TextureUsageFlags::Storage |
                    TextureUsageFlags::TransferSrc,
                .debugName = "PathTracer.TonemapLDR"
                });

            PathTracePassData pathTrace{};
            pathTrace.outputAccumulation = accumulationTexture;
            pathTrace.pipeline = makePipelineId("path_trace");

            pathTraceNode =
                builder.addGraphNode(
                    pathTrace,
                    GraphNodeType::Compute,
                    QueueType::Compute);

            builder.write(
                pathTraceNode,
                accumulationTexture,
                ResourceUsage::StorageTexture);
            builder.read(
                pathTraceNode,
                environmentCubemap,
                ResourceUsage::SampledTexture);

            TonemapPassData tonemap{};
            tonemap.inputHDR = accumulationTexture;
            tonemap.outputBackBuffer = tonemapTexture;
            tonemap.pipeline = makePipelineId("path_trace_tonemap");

            tonemapNode =
                builder.addGraphNode(
                    tonemap,
                    GraphNodeType::Compute,
                    QueueType::Compute);

            builder.read(
                tonemapNode,
                accumulationTexture,
                ResourceUsage::SampledTexture);

            builder.write(
                tonemapNode,
                tonemapTexture,
                ResourceUsage::StorageTexture);

            builder.setResourceSemantic(
                tonemapTexture,
                GraphResourceSemantic::PathTraceTonemap);

            // Swapchain resources require a graphics/direct queue on D3D12.
            // Ordinary runtime copies retain the transfer-queue default.
            copyNode = builder
                .addTransferPass("PathTracer.CopyToBackBuffer")
                .copy(tonemapTexture, backBuffer)
                .queue(QueueType::Graphics);

            presentNode =
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
