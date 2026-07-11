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
        resources.instanceBounds = builder.createBuffer({
            .size = MaxGpuDrivenDraws * sizeof(GpuInstanceBounds),
            .usage = BufferUsageFlags::Storage,
            .memoryUsage = ResourceMemoryUsage::GpuOnly,
            .debugName = "GpuDriven.InstanceBounds" });
        resources.drawInputs = builder.createBuffer({
            .size = MaxGpuDrivenDraws * sizeof(GpuDrawInput),
            .usage = BufferUsageFlags::Storage,
            .memoryUsage = ResourceMemoryUsage::GpuOnly,
            .debugName = "GpuDriven.DrawInputs" });
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
        resources.indirectArguments = builder.createBuffer({
            .size = MaxGpuDrivenDraws * sizeof(GpuIndexedIndirectArguments),
            .usage = BufferUsageFlags::Storage | BufferUsageFlags::Indirect,
            .memoryUsage = ResourceMemoryUsage::GpuOnly,
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
