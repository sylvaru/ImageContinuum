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
        GraphResourceId accumulationTexture;
        GraphResourceId tonemapTexture;
        GraphNodeId pathTraceNode = InvalidGraphNodeId;
        GraphNodeId tonemapNode = InvalidGraphNodeId;
        GraphNodeId copyNode = InvalidGraphNodeId;
        GraphNodeId presentNode = InvalidGraphNodeId;

        void buildFrameGraph(FrameGraphBuilder& builder) override
        {
            backBuffer = builder.importTexture({
                .type = ImportedResource::Swapchain,
                .initialUsage = ResourceUsage::Present
                });

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

            TransferPassData copy{};
            copy.name = "PathTracer.CopyToBackBuffer";
            copy.source = tonemapTexture;
            copy.destination = backBuffer;

            copyNode =
                builder.addGraphNode(
                    copy,
                    GraphNodeType::Transfer,
                    QueueType::Transfer);

            builder.read(
                copyNode,
                tonemapTexture,
                ResourceUsage::TransferSrc);

            builder.write(
                copyNode,
                backBuffer,
                ResourceUsage::TransferDst);

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
