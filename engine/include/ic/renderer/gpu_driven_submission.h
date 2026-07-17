#pragma once

#include <cstdint>

#include "ic/renderer/frame_graph/frame_graph_builder.h"
#include "ic/renderer/renderer_gpu_assets.h"

namespace ic
{
    inline constexpr uint32_t MaxGpuDrivenDraws =
        ClusteredForwardMaxGpuCullInstances;
    inline constexpr uint32_t MaxGpuDrivenBins = 4096;

    struct GpuDrivenGraphResources
    {
        GraphResourceId instanceBounds = InvalidGraphResourceId;
        GraphResourceId drawInputs = InvalidGraphResourceId;
        GraphResourceId visibleInstances = InvalidGraphResourceId;
        GraphResourceId visibleCount = InvalidGraphResourceId;
        GraphResourceId indirectArguments = InvalidGraphResourceId;
        GraphResourceId drawMetadata = InvalidGraphResourceId;
        GraphResourceId binCounts = InvalidGraphResourceId;
        GraphResourceId cullClassification = InvalidGraphResourceId;
        GraphResourceId cullStats = InvalidGraphResourceId;
        GraphResourceId cullStatsReadback = InvalidGraphResourceId;
    };

    inline GpuDrivenGraphResources declareGpuDrivenResources(
        FrameGraphBuilder& builder,
        bool diagnosticsEnabled = false)
    {
        GpuDrivenGraphResources resources{};

        // Cull inputs are CPU-uploaded each frame; the registry hands out a
        // per-frame-slot mapped instance so the backend memcpys straight into
        // the graph-owned buffer (no separate backend upload buffer).
        resources.instanceBounds = builder.createBuffer({
            .size = MaxGpuDrivenDraws * sizeof(GpuInstanceBounds),
            .usage = BufferUsageFlags::Storage,
            .memoryUsage = ResourceMemoryUsage::CpuToGpu,
            .mappedAtCreation = true,
            .debugName = "GpuDriven.InstanceBounds" });
        resources.drawInputs = builder.createBuffer({
            .size = MaxGpuDrivenDraws * sizeof(GpuDrawInput),
            .usage = BufferUsageFlags::Storage,
            .memoryUsage = ResourceMemoryUsage::CpuToGpu,
            .mappedAtCreation = true,
            .debugName = "GpuDriven.DrawInputs" });

        // Cull outputs are GPU-only; the graph's barriers own their state and
        // the queue/UAV ordering (no manual backend state tracking).
        resources.visibleInstances = builder.createBuffer({
            .size = MaxGpuDrivenDraws * sizeof(uint32_t),
            .usage = BufferUsageFlags::Storage,
            .memoryUsage = ResourceMemoryUsage::GpuOnly,
            .debugName = "GpuDriven.VisibleInstances" });
        resources.visibleCount = builder.createBuffer({
            .size = sizeof(uint32_t),
            .usage = BufferUsageFlags::Storage | BufferUsageFlags::TransferSrc,
            .memoryUsage = ResourceMemoryUsage::GpuOnly,
            .debugName = "GpuDriven.VisibleCount" });
        // Native command stride is backend-defined: declare by element count and
        // let each backend's registry size it (DX12 36 bytes w/ root constants,
        // Vulkan 20-byte VkDrawIndexedIndirectCommand). See BufferDesc.
        resources.indirectArguments = builder.createBuffer({
            .usage = BufferUsageFlags::Storage | BufferUsageFlags::Indirect,
            .memoryUsage = ResourceMemoryUsage::GpuOnly,
            .elementCount = MaxGpuDrivenDraws,
            .debugName = "GpuDriven.IndirectArguments" });
        resources.drawMetadata = builder.createBuffer({
            .size = MaxGpuDrivenDraws * sizeof(GpuDrawMetadata),
            .usage = BufferUsageFlags::Storage,
            .memoryUsage = ResourceMemoryUsage::GpuOnly,
            .debugName = "GpuDriven.DrawMetadata" });
        resources.binCounts = builder.createBuffer({
            .size = MaxGpuDrivenBins * sizeof(uint32_t),
            .usage = BufferUsageFlags::Storage | BufferUsageFlags::Indirect,
            .memoryUsage = ResourceMemoryUsage::GpuOnly,
            .debugName = "GpuDriven.BinCounts" });
        resources.cullClassification = builder.createBuffer({
            .size = diagnosticsEnabled
                ? MaxGpuDrivenDraws * sizeof(uint32_t)
                : sizeof(uint32_t),
            .usage = BufferUsageFlags::Storage,
            .memoryUsage = ResourceMemoryUsage::GpuOnly,
            .debugName = "GpuDriven.CullClassification" });
        resources.cullStats = builder.createBuffer({
            .size = sizeof(GpuCullStats),
            .usage =
                BufferUsageFlags::Storage |
                BufferUsageFlags::TransferSrc,
            .memoryUsage = ResourceMemoryUsage::GpuOnly,
            .debugName = "GpuDriven.CullStats" });
        if (diagnosticsEnabled)
        {
            resources.cullStatsReadback = builder.createBuffer({
                .size = sizeof(GpuCullStats),
                .usage = BufferUsageFlags::TransferDst,
                .memoryUsage = ResourceMemoryUsage::GpuToCpu,
                .mappedAtCreation = true,
                .debugName = "GpuDriven.CullStatsReadback" });
        }

        builder.setResourceSemantic(
            resources.instanceBounds,
            GraphResourceSemantic::GpuDrivenInstanceBounds);
        builder.setResourceSemantic(
            resources.drawInputs,
            GraphResourceSemantic::GpuDrivenDrawInputs);
        builder.setResourceSemantic(
            resources.visibleInstances,
            GraphResourceSemantic::GpuDrivenVisibleInstances);
        builder.setResourceSemantic(
            resources.visibleCount,
            GraphResourceSemantic::GpuDrivenVisibleCount);
        builder.setResourceSemantic(
            resources.indirectArguments,
            GraphResourceSemantic::GpuDrivenIndirectArguments);
        builder.setResourceSemantic(
            resources.drawMetadata,
            GraphResourceSemantic::GpuDrivenDrawMetadata);
        builder.setResourceSemantic(
            resources.binCounts,
            GraphResourceSemantic::GpuDrivenBinCounts);
        builder.setResourceSemantic(
            resources.cullClassification,
            GraphResourceSemantic::GpuDrivenCullClassification);
        builder.setResourceSemantic(
            resources.cullStats,
            GraphResourceSemantic::GpuDrivenCullStats);
        if (resources.cullStatsReadback != InvalidGraphResourceId)
        {
            builder.setResourceSemantic(
                resources.cullStatsReadback,
                GraphResourceSemantic::GpuDrivenCullStatsReadback);
        }
        return resources;
    }

    inline GraphNodeId addGpuDrivenCullPasses(
        FrameGraphBuilder& builder,
        const GpuDrivenGraphResources& resources,
        GraphResourceId previousHiZ = InvalidGraphResourceId)
    {
        builder.addComputePass("ResetGpuDrivenCounts")
            .pipeline("reset_visible_instance_count")
            .dispatch((MaxGpuDrivenBins + 63u) / 64u, 1, 1)
            .write(resources.visibleCount, ResourceUsage::StorageBuffer)
            .write(resources.binCounts, ResourceUsage::StorageBuffer)
            .write(resources.cullStats, ResourceUsage::StorageBuffer);

        auto cull = builder.addComputePass("GPUFrustumCullAndBuildCommands")
            .pipeline("gpu_frustum_cull")
            .dispatch((MaxGpuDrivenDraws + 63u) / 64u, 1, 1)
            .read(resources.instanceBounds, ResourceUsage::StorageBuffer)
            .read(resources.drawInputs, ResourceUsage::StorageBuffer)
            .write(resources.visibleInstances, ResourceUsage::StorageBuffer)
            .write(resources.visibleCount, ResourceUsage::StorageBuffer)
            .write(resources.indirectArguments, ResourceUsage::StorageBuffer)
            .write(resources.drawMetadata, ResourceUsage::StorageBuffer)
            .write(resources.binCounts, ResourceUsage::StorageBuffer)
            .write(resources.cullClassification, ResourceUsage::StorageBuffer)
            .write(resources.cullStats, ResourceUsage::StorageBuffer);
        if (previousHiZ != InvalidGraphResourceId)
        {
            builder.readPrevious(
                cull, previousHiZ, ResourceUsage::SampledTexture);
        }
        return cull;
    }

    inline void addGpuDrivenOcclusionValidationPasses(
        FrameGraphBuilder& builder,
        const GpuDrivenGraphResources& resources,
        GraphResourceId currentHiZ,
        QueueType validationQueue)
    {
        if (resources.cullStatsReadback == InvalidGraphResourceId)
        {
            return;
        }

        // Diagnostics only: reads the cull classification and the current Hi-Z
        // and writes its own stats buffer, so it shares nothing mutable with
        // concurrent graphics work and is safe on the async queue. Whether it
        // is worth a queue of its own is the scheduler's call.
        builder.addComputePass("ValidateGpuOcclusion", validationQueue)
            .pipeline("gpu_occlusion_validate")
            .dispatch((MaxGpuDrivenDraws + 63u) / 64u, 1, 1)
            .read(resources.instanceBounds, ResourceUsage::StorageBuffer)
            .read(resources.cullClassification, ResourceUsage::StorageBuffer)
            .read(currentHiZ, ResourceUsage::SampledTexture)
            .write(resources.cullStats, ResourceUsage::StorageBuffer)
            .asyncEligible();

        builder.addTransferPass("ReadbackGpuCullStats")
            .copy(resources.cullStats, resources.cullStatsReadback)
            .queue(QueueType::Transfer);
    }

    inline void readGpuDrivenStream(
        FrameGraphBuilder& builder,
        GraphNodeId node,
        const GpuDrivenGraphResources& resources)
    {
        builder.read(node, resources.indirectArguments,
            ResourceUsage::IndirectArgument);
        builder.read(node, resources.binCounts,
            ResourceUsage::IndirectArgument);
        builder.read(node, resources.drawMetadata,
            ResourceUsage::StorageBuffer);
    }
}
