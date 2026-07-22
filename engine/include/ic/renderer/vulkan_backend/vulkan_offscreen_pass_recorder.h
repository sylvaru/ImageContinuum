#pragma once

// Recorders for the Vulkan offscreen compute passes: HDR equirect->cubemap
// conversion, the path tracer, and its tonemap. Each owns its GPU constants
// build, native bindings, and dispatch. The backend owns all lifetime
// (ensure/destroy/upload, descriptor-set allocation + update via the
// update*Descriptors methods), cross-frame accumulation state, and the
// state-tracked image-layout transitions, resolving the descriptor set + the
// constants buffer into the *Inputs structs below.

#include "ic/renderer/vulkan_backend/vulkan_pass_recorders.h"
#include "ic/renderer/vulkan_backend/vulkan_pipeline_manager.h"
#include "ic/renderer/vulkan_backend/vulkan_resource_allocator.h"

#include <vulkan/vulkan.h>
#include <cstdint>

namespace ic
{
    // HDR equirectangular source -> cubemap face conversion. The backend has
    // already allocated/updated the descriptor set and transitioned the cubemap
    // to GENERAL; the recorder binds and dispatches the 6 faces.
    void recordEnvironmentConvert(
        const VulkanPassContext& ctx,
        const VulkanComputePipeline& pipeline,
        VkDescriptorSet descriptorSet,
        uint32_t cubemapSize);

    struct VulkanPathTraceInputs
    {
        const VulkanComputePipeline* pipeline = nullptr;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        VulkanBuffer* constants = nullptr;

        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t frameIndex = 0;
        uint32_t accumulatedSampleCount = 0;
        float exposure = 1.0f;
        bool resetAccumulation = false;
        uint32_t sceneVertexCount = 0;
        uint32_t sceneMaterialCount = 0;
        uint32_t sceneTriangleCount = 0;
        uint32_t sceneBvhNodeCount = 0;
        uint32_t firstEmissiveTriangleIndex = 0;
        uint32_t emissiveTriangleCount = 0;
        bool environmentReady = false;
        float environmentIntensity = 0.0f;
        float environmentExposure = 0.0f;
    };

    void recordPathTrace(
        const VulkanPassContext& ctx,
        const VulkanPathTraceInputs& inputs);

    struct VulkanTonemapInputs
    {
        const VulkanComputePipeline* pipeline = nullptr;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        VulkanBuffer* constants = nullptr;
        uint32_t width = 0;
        uint32_t height = 0;
        float exposure = 1.0f;
    };

    void recordTonemap(
        const VulkanPassContext& ctx,
        const VulkanTonemapInputs& inputs);
}
