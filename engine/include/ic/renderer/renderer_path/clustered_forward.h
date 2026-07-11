// ic/renderer/renderer_path/forward_path.h
#pragma once
#include "renderer_path.h"
#include "ic/renderer/frame_graph/frame_graph_types.h"
#include "ic/renderer/frame_graph/frame_graph_builder.h"
#include "ic/renderer/gpu_driven_submission.h"


namespace ic
{
    class ClusteredForwardRendererPath : public RendererPath
    {
    public:
        GraphResourceId backBuffer;

        GraphResourceId sceneDepth;
        GraphResourceId depthPyramid;

        GraphResourceId clusterBoundsBuffer;
        GraphResourceId clusterLightGridBuffer;
        GraphResourceId clusterLightIndexBuffer;
        GraphResourceId clusterLightCounterBuffer;
        GraphResourceId visibleLightBuffer;
        GpuDrivenGraphResources gpuDriven;

        GraphResourceId environmentCubemap;

        void buildFrameGraph(
            RendererPathContext& ctx,
            FrameGraphBuilder& builder) override
        {
            uint32_t renderWidth = ctx.renderExtent.width;
            uint32_t renderHeight = ctx.renderExtent.height;

            constexpr uint32_t clusterTileSizeX = ClusteredForwardTileSizeX;
            constexpr uint32_t clusterTileSizeY = ClusteredForwardTileSizeY;
            constexpr uint32_t clusterSliceCountZ = ClusteredForwardSliceCountZ;

            uint32_t clusterCountX =
                (renderWidth + clusterTileSizeX - 1) / clusterTileSizeX;

            uint32_t clusterCountY =
                (renderHeight + clusterTileSizeY - 1) / clusterTileSizeY;

            uint32_t clusterCount =
                clusterCountX * clusterCountY * clusterSliceCountZ;

            constexpr uint32_t maxLightsPerCluster =
                ClusteredForwardMaxLightsPerCluster;
            uint32_t maxClusterLightRefs =
                clusterCount * maxLightsPerCluster;

            constexpr uint32_t threadsPerGroup = 64;

            const uint32_t clusterDispatchGroups =
                (clusterCount + threadsPerGroup - 1) / threadsPerGroup;

            const uint32_t hiZMipCount =
                1 + static_cast<uint32_t>(
                    std::floor(std::log2(std::max(renderWidth, renderHeight))));

            // Imported resources

            backBuffer = builder.importTexture({
                .type = ImportedResource::Swapchain,
                .initialUsage = ResourceUsage::Present
                });

            // Per frame render resources
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
                .debugName = "ClusteredForward.SceneDepth"
                });

            depthPyramid = builder.createTexture({
                .width = renderWidth,
                .height = renderHeight,
                .depth = 1,
                .mipLevels = hiZMipCount,
                .arrayLayers = 1,
                .format = TextureFormat::R32_Float,
                .usage =
                    TextureUsageFlags::Sampled |
                    TextureUsageFlags::Storage,
                .debugName = "ClusteredForward.DepthPyramid"
                });

            clusterBoundsBuffer = builder.createBuffer({
                .size = clusterCount * sizeof(GpuClusterBounds),
                .usage = BufferUsageFlags::Storage,
                .memoryUsage = ResourceMemoryUsage::GpuOnly,
                .mappedAtCreation = false,
                .debugName = "ClusteredForward.ClusterBounds"
                });

            clusterLightGridBuffer = builder.createBuffer({
                .size = clusterCount * sizeof(GpuClusterLightGrid),
                .usage = BufferUsageFlags::Storage,
                .memoryUsage = ResourceMemoryUsage::GpuOnly,
                .mappedAtCreation = false,
                .debugName = "ClusteredForward.ClusterLightGrid"
                });

            clusterLightIndexBuffer = builder.createBuffer({
                .size = maxClusterLightRefs * sizeof(uint32_t),
                .usage = BufferUsageFlags::Storage,
                .memoryUsage = ResourceMemoryUsage::GpuOnly,
                .mappedAtCreation = false,
                .debugName = "ClusteredForward.ClusterLightIndices"
                });

            clusterLightCounterBuffer = builder.createBuffer({
                .size = sizeof(uint32_t),
                .usage = BufferUsageFlags::Storage,
                .memoryUsage = ResourceMemoryUsage::GpuOnly,
                .mappedAtCreation = false,
                .debugName = "ClusteredForward.ClusterLightCounter"
                });

            visibleLightBuffer = builder.createBuffer({
                .size = ClusteredForwardMaxVisibleLights * sizeof(GpuVisibleLight),
                .usage = BufferUsageFlags::Storage,
                .memoryUsage = ResourceMemoryUsage::GpuOnly,
                .mappedAtCreation = false,
                .debugName = "ClusteredForward.VisibleLights"
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

            // Environment conversion
            // Fine for now, but eventually this should only run when the environment changes

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

            gpuDriven = declareGpuDrivenResources(builder);
            addGpuDrivenCullPasses(builder, gpuDriven);

            // Frustum culling is deliberately before depth. A previous-frame
            // Hi-Z input can be inserted into the shared cull stage later.

            auto depthPrepassNode =
                builder.addGraphicsPass("DepthPrepass")
                .pipeline("depth_prepass")
                .drawList(DrawListKind::SceneGeometry)
                .depthLoadOp(AttachmentLoadOp::Clear)
                .depth(sceneDepth);
            readGpuDrivenStream(builder, depthPrepassNode, gpuDriven);


            auto hiZNode =
                builder.addComputePass("BuildHiZ")
                .pipeline("build_hiz")
                .dispatch((renderWidth + 7) / 8, (renderHeight + 7) / 8, 1)
                .read(sceneDepth, ResourceUsage::SampledTexture)
                .write(depthPyramid, ResourceUsage::StorageTexture);

            // Build clusters
            //
            // This can eventually be cached and rebuilt only when projection,
            // render extent, tile size, or z slice count changes

            auto clusterBuildNode =
                builder.addComputePass("BuildClusters")
                .pipeline("cluster_build")
                .dispatch(clusterDispatchGroups, 1, 1)
                .write(clusterBoundsBuffer, ResourceUsage::StorageBuffer);

            // Reset global append counter before light culling.

            auto resetClusterCounterNode =
                builder.addComputePass("ResetClusterLightCounter")
                .pipeline("reset_cluster_light_counter")
                .dispatch(1, 1, 1)
                .write(clusterLightCounterBuffer, ResourceUsage::StorageBuffer);

            // Cull lights into clusters.
            //
            // First version: one compute invocation per cluster.
            // Each cluster loops over all visible lights and appends matching light
            // indices into clusterLightIndexBuffer using clusterLightCounterBuffer

            auto lightCullNode =
                builder.addComputePass("CullLightsToClusters")
                .pipeline("cluster_light_cull")
                .dispatch(clusterDispatchGroups, 1, 1)
                .read(clusterBoundsBuffer, ResourceUsage::StorageBuffer)
                .read(visibleLightBuffer, ResourceUsage::StorageBuffer)
                .read(clusterLightCounterBuffer, ResourceUsage::StorageBuffer)
                .write(clusterLightGridBuffer, ResourceUsage::StorageBuffer)
                .write(clusterLightIndexBuffer, ResourceUsage::StorageBuffer)
                .write(clusterLightCounterBuffer, ResourceUsage::StorageBuffer);


            // Opaque clustered forward pass
            auto opaqueNode =
                builder.addGraphicsPass("ClusteredForwardOpaque")
                .pipeline("clustered_forward_opaque")
                .drawList(DrawListKind::SceneGeometry)
                .colorLoadOp(AttachmentLoadOp::Clear)
                .depthLoadOp(AttachmentLoadOp::Load)
                .color(backBuffer)
                .depth(sceneDepth);
            readGpuDrivenStream(builder, opaqueNode, gpuDriven);

            builder.read(
                opaqueNode,
                clusterLightGridBuffer,
                ResourceUsage::StorageBuffer);

            builder.read(
                opaqueNode,
                clusterLightIndexBuffer,
                ResourceUsage::StorageBuffer);

            builder.read(
                opaqueNode,
                visibleLightBuffer,
                ResourceUsage::StorageBuffer);

            builder.read(
                opaqueNode,
                environmentCubemap,
                ResourceUsage::SampledTexture);

            // Skybox
            auto skyboxNode =
                builder.addGraphicsPass("Skybox")
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

            // Present

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
