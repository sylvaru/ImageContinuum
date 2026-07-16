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
    };

    inline GpuDrivenGraphResources declareGpuDrivenResources(
        FrameGraphBuilder& builder)
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
        return resources;
    }

    inline GraphNodeId addGpuDrivenCullPasses(
        FrameGraphBuilder& builder,
        const GpuDrivenGraphResources& resources)
    {
        builder.addComputePass("ResetGpuDrivenCounts")
            .pipeline("reset_visible_instance_count")
            .dispatch((MaxGpuDrivenBins + 63u) / 64u, 1, 1)
            .write(resources.visibleCount, ResourceUsage::StorageBuffer)
            .write(resources.binCounts, ResourceUsage::StorageBuffer);

        auto cull = builder.addComputePass("GPUFrustumCullAndBuildCommands")
            .pipeline("gpu_frustum_cull")
            .dispatch((MaxGpuDrivenDraws + 63u) / 64u, 1, 1)
            .read(resources.instanceBounds, ResourceUsage::StorageBuffer)
            .read(resources.drawInputs, ResourceUsage::StorageBuffer)
            .write(resources.visibleInstances, ResourceUsage::StorageBuffer)
            .write(resources.visibleCount, ResourceUsage::StorageBuffer)
            .write(resources.indirectArguments, ResourceUsage::StorageBuffer)
            .write(resources.drawMetadata, ResourceUsage::StorageBuffer)
            .write(resources.binCounts, ResourceUsage::StorageBuffer);
        return cull;
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
