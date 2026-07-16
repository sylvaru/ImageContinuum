#pragma once

// Recorders for the Vulkan main render-path passes: the clustered light-culling
// compute binding, the forward/depth scene pass, and the skybox. Each owns its
// native bindings, viewport/scissor, and draw/dispatch. The backend owns
// lifetime (ensure/prepare, descriptor-set update), dynamic-rendering
// begin/end, attachment setup, and the state-tracked transitions, resolving the
// per-frame descriptor set / GPU-scene handles into the *Inputs structs.
//
// Vulkan binds a single unified per-frame descriptor set (built by the backend's
// updateFrameDescriptors), so the clustered and generic scene paths share one
// binding, unlike DX12's many root parameters.

#include "ic/renderer/vulkan_backend/vulkan_pass_recorders.h"
#include "ic/renderer/vulkan_backend/vulkan_pipeline_manager.h"
#include "ic/renderer/vulkan_backend/vulkan_gpu_scene.h"
#include "ic/renderer/vulkan_backend/vulkan_resource_allocator.h"

#include <vulkan/vulkan.h>
#include <span>

namespace ic
{
    // Binds the clustered light-culling compute pass: the unified per-frame
    // descriptor set (frame constants + cluster + light buffers). The caller has
    // already bound the pipeline and issues the dispatch.
    void recordClusteredForwardCompute(
        const VulkanPassContext& ctx,
        const VulkanComputePipeline& pipeline,
        VkDescriptorSet descriptorSet);

    struct VulkanForwardSceneInputs
    {
        const VulkanGraphicsPipeline* pipeline = nullptr;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        bool useGpuDriven = false;
        VulkanIndirectDrawStream indirectStream = {};
        std::span<const VulkanGpuScene::DrawItem> draws = {};
        std::span<const VulkanGpuScene::GeometryBin> geometryBins = {};
        VulkanResolveNativeModelFn resolveNativeModel = {};
    };

    void recordForwardScene(
        const VulkanPassContext& ctx,
        const VulkanForwardSceneInputs& inputs);

    struct VulkanSkyboxInputs
    {
        const VulkanGraphicsPipeline* pipeline = nullptr;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        VulkanBuffer* constants = nullptr;
    };

    void recordSkybox(
        const VulkanPassContext& ctx,
        const VulkanSkyboxInputs& inputs);
}
