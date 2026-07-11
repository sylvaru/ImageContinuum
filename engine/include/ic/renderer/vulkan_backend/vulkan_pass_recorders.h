#pragma once

#include "ic/renderer/vulkan_backend/vulkan_graph_resource_registry.h"
#include "ic/renderer/vulkan_backend/vulkan_gpu_scene.h"
#include "ic/renderer/vulkan_backend/vulkan_pipeline_manager.h"
#include "ic/renderer/frame_graph/compiled_graph_plan.h"

#include <functional>
#include <span>
#include <vulkan/vulkan.h>

namespace ic
{
    // Lightweight, API-specific pass context handed to the Vulkan pass
    // recorders. Carries the native command buffer, executing node, graph
    // resource registry and device. It does NOT own the swapchain, frame
    // lifecycle or queue submission; the backend performs resource
    // "ensure"/prepare before invoking a recorder. Passed by const&.
    struct VulkanPassContext
    {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        const CompiledGraphPlan* plan = nullptr;
        const ExecutionNode* node = nullptr;
        VulkanGraphResourceRegistry* resources = nullptr;
        VkDevice device = VK_NULL_HANDLE;
    };

    struct VulkanHiZInputs
    {
        GraphResourceId hiZId = InvalidGraphResourceId;
        GraphResourceId sceneDepthId = InvalidGraphResourceId;
        // Per-frame descriptor pool (created lazily; owned by the caller's
        // frame scene resources).
        VkDescriptorPool* hiZPool = nullptr;
        VkBuffer frameConstants = VK_NULL_HANDLE;
        GraphResourceId* hiZDebugResourceOut = nullptr;
    };

    // Native view of the GPU-driven cull buffers + inputs. The backend fills
    // this from VulkanGpuScene, which owns the buffers.
    struct VulkanCullBuffers
    {
        VkDescriptorPool* gpuCullPool = nullptr;
        VkBuffer frameConstants = VK_NULL_HANDLE;
        VkBuffer instanceBounds = VK_NULL_HANDLE;
        VkDeviceSize instanceBoundsSize = 0;
        VkBuffer visibleInstances = VK_NULL_HANDLE;
        VkDeviceSize visibleInstancesSize = 0;
        VkBuffer visibleInstanceCount = VK_NULL_HANDLE;
        VkDeviceSize visibleInstanceCountSize = 0;
        VkBuffer drawInputs = VK_NULL_HANDLE;
        VkDeviceSize drawInputsSize = 0;
        VkBuffer indirectArguments = VK_NULL_HANDLE;
        VkDeviceSize indirectArgumentsSize = 0;
        VkBuffer drawMetadata = VK_NULL_HANDLE;
        VkDeviceSize drawMetadataSize = 0;
        VkBuffer binCounts = VK_NULL_HANDLE;
        VkDeviceSize binCountsSize = 0;
    };

    // Records the Hi-Z depth-pyramid mip chain. Returns true when it recorded.
    bool recordHiZPyramid(
        const VulkanPassContext& ctx,
        const VulkanComputePipeline& pipeline,
        const VulkanHiZInputs& inputs);

    // Records the GPU frustum-cull descriptor binding (the dispatch is issued by
    // the caller). Returns false if descriptor allocation fails.
    bool recordGpuFrustumCull(
        const VulkanPassContext& ctx,
        const VulkanComputePipeline& pipeline,
        const VulkanCullBuffers& buffers);

    // Native indirect-draw stream consumed when useGpuDriven is set.
    struct VulkanIndirectDrawStream
    {
        VkBuffer indirectArguments = VK_NULL_HANDLE;
        VkBuffer binCounts = VK_NULL_HANDLE;
    };

    // Resolves a scene draw's AssetHandle to its uploaded native model
    // (vertex/index buffers). Draws/bins are grouped contiguously by model
    // (PreparedGpuScene sorts by model), so this is called once per bin
    // transition, not once per draw.
    using VulkanResolveNativeModelFn =
        std::function<VulkanUploadedModel*(AssetHandle)>;

    // Records the per-draw vertex/index-buffer binding + push-constant +
    // DrawIndexed submission loop for a prepared scene draw list, or the
    // vkCmdDrawIndexedIndirectCount-per-bin path when useGpuDriven is set.
    // This is the draw-submission body shared by the depth prepass and the
    // forward scene pass alike — both pipelines route scene geometry through
    // this same recorder, differing only in bound descriptor set/pixel
    // output, which the caller sets up beforehand.
    void recordSceneGeometryDraws(
        VkCommandBuffer cmd,
        VkPipelineLayout pipelineLayout,
        std::span<const VulkanGpuScene::DrawItem> draws,
        std::span<const VulkanGpuScene::GeometryBin> geometryBins,
        bool useGpuDriven,
        const VulkanIndirectDrawStream& indirectStream,
        const VulkanResolveNativeModelFn& resolveNativeModel);
}
