#pragma once

#include "ic/renderer/vulkan_backend/vulkan_graph_resource_registry.h"
#include "ic/renderer/vulkan_backend/vulkan_gpu_scene.h"
#include "ic/renderer/vulkan_backend/vulkan_pipeline_manager.h"
#include "ic/renderer/frame_graph/compiled_graph_plan.h"

#include <cstdint>
#include <functional>
#include <span>
#include <vulkan/vulkan.h>

namespace ic
{
    struct FrameContext;
    struct SceneRenderView;
    class VulkanResourceAllocator;
    class VulkanPipelineManager;
    class VulkanGpuScene;

    // Lightweight, API-specific pass context handed to the Vulkan pass
    // recorders. Carries the native command buffer, executing node/plan,
    // per-frame + scene views, and narrow references to the SHARED subsystems a
    // recorder reads to resolve graph resources, upload constants, and bind
    // pipelines. It does NOT own the swapchain lifecycle, queue submission, or
    // any backend-private per-technique resource struct (those the backend
    // resolves into a pass-specific *Inputs struct, performing ensure/prepare and
    // state-tracked transitions itself). Passed by const&.
    struct VulkanPassContext
    {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        const CompiledGraphPlan* plan = nullptr;
        const ExecutionNode* node = nullptr;
        const FrameContext* frame = nullptr;
        const SceneRenderView* scene = nullptr;
        VulkanGraphResourceRegistry* resources = nullptr;
        VkDevice device = VK_NULL_HANDLE;
        VulkanResourceAllocator* allocator = nullptr;
        VulkanPipelineManager* pipelines = nullptr;
        VulkanGpuScene* gpuScene = nullptr;
        VkExtent2D surfaceExtent = {};
    };

    // Native, already-resolved source/destination for a transfer-copy pass. The
    // backend resolves both (imported swapchain, backend-private path-trace
    // targets) and applies any state-tracked transition first; the recorder owns
    // the copy command.
    struct VulkanTransferCopy
    {
        bool isBuffer = false;
        VkBuffer sourceBuffer = VK_NULL_HANDLE;
        VkBuffer destinationBuffer = VK_NULL_HANDLE;
        VkDeviceSize bufferSize = 0;
        VkImage sourceImage = VK_NULL_HANDLE;
        VkImage destinationImage = VK_NULL_HANDLE;
        uint32_t width = 0;
        uint32_t height = 0;
    };

    void recordTransferCopy(
        VkCommandBuffer cmd,
        const VulkanTransferCopy& copy);

    // Diagnostic storage-buffer compute pass. The backend has bound the pipeline
    // and ensured the resources + descriptor set; the recorder binds the set,
    // dispatches, and issues the trailing buffer barrier.
    struct VulkanComputeStorageBufferTest
    {
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceSize bufferSize = 0;
        uint32_t groupCountX = 1;
        uint32_t groupCountY = 1;
        uint32_t groupCountZ = 1;
    };

    void recordComputeStorageBufferTest(
        const VulkanPassContext& ctx,
        const VulkanComputeStorageBufferTest& test);

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
    // forward scene pass alike. Both pipelines route scene geometry through
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
