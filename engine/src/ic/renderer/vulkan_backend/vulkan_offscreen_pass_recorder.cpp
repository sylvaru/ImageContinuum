#include "ic/renderer/vulkan_backend/vulkan_offscreen_pass_recorder.h"

#include "ic/scene/scene_render_view.h"
#include "ic/scene/scene_components.h"
#include "ic/renderer/renderer_common/renderer_util.h"

#include <cstring>

namespace ic
{
    void recordEnvironmentConvert(
        const VulkanPassContext& ctx,
        const VulkanComputePipeline& pipeline,
        VkDescriptorSet descriptorSet,
        uint32_t cubemapSize)
    {
        VkCommandBuffer cmd = ctx.cmd;
        vkCmdBindPipeline(
            cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.pipeline);
        vkCmdBindDescriptorSets(
            cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.pipelineLayout,
            0,
            1,
            &descriptorSet,
            0,
            nullptr);
        vkCmdDispatch(
            cmd,
            (cubemapSize + 7u) / 8u,
            (cubemapSize + 7u) / 8u,
            6u);
    }

    void recordPathTrace(
        const VulkanPassContext& ctx,
        const VulkanPathTraceInputs& inputs)
    {
        const SceneRenderView& scene = *ctx.scene;

        PathTraceConstants constants{};
        constants.renderWidth = inputs.width;
        constants.renderHeight = inputs.height;
        constants.frameIndex = inputs.frameIndex;
        constants.accumulatedSampleCount = inputs.accumulatedSampleCount;
        constants.exposure = inputs.exposure;
        constants.resetAccumulation = inputs.resetAccumulation ? 1u : 0u;
        constants.maxBounces = configuredPathTraceMaxBounces();
        constants.samplesPerPixel = DefaultPathTraceSamplesPerPixel;
        constants.sceneVertexCount = inputs.sceneVertexCount;
        constants.sceneMaterialCount = inputs.sceneMaterialCount;
        constants.sceneTriangleCount = inputs.sceneTriangleCount;
        constants.sceneBvhNodeCount = inputs.sceneBvhNodeCount;
        constants.sceneEmissiveTriangleIndex = inputs.firstEmissiveTriangleIndex;
        constants.sceneEmissiveTriangleCount = inputs.emissiveTriangleCount;
        constants.referenceMode = configuredPathTraceReferenceMode();
        constants.useSceneGeometry =
            inputs.sceneTriangleCount != 0u &&
            inputs.sceneBvhNodeCount != 0u
                ? 1u
                : 0u;
        constants.environmentEnabled = inputs.environmentReady ? 1u : 0u;
        constants.environmentIntensity = inputs.environmentIntensity;
        constants.environmentExposure = inputs.environmentExposure;
        fillPathTraceCameraConstants(
            scene.camera,
            inputs.width,
            inputs.height,
            constants);
        for (const SceneLightRenderItem& light : scene.lights)
        {
            if (light.type != LightType::Point ||
                constants.pointLightCount >= MaxPathTracePointLights)
            {
                continue;
            }

            const uint32_t lightIndex = constants.pointLightCount++;
            constants.pointLightPositionRange[lightIndex] =
                glm::vec4(light.position, light.range);
            constants.pointLightColorIntensity[lightIndex] =
                glm::vec4(light.color, light.intensity);
        }

        std::memcpy(
            inputs.constants->mapped,
            &constants,
            sizeof(constants));
        ctx.allocator->flush(*inputs.constants, 0, sizeof(constants));

        VkCommandBuffer cmd = ctx.cmd;
        vkCmdBindPipeline(
            cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            inputs.pipeline->pipeline);
        vkCmdBindDescriptorSets(
            cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            inputs.pipeline->pipelineLayout,
            0,
            1,
            &inputs.descriptorSet,
            0,
            nullptr);

        const uint32_t groupCountX = (inputs.width + 7u) / 8u;
        const uint32_t groupCountY = (inputs.height + 7u) / 8u;
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);
    }

    void recordTonemap(
        const VulkanPassContext& ctx,
        const VulkanTonemapInputs& inputs)
    {
        TonemapConstants constants{};
        constants.renderWidth = inputs.width;
        constants.renderHeight = inputs.height;
        constants.exposure = inputs.exposure;

        std::memcpy(
            inputs.constants->mapped,
            &constants,
            sizeof(constants));
        ctx.allocator->flush(*inputs.constants, 0, sizeof(constants));

        VkCommandBuffer cmd = ctx.cmd;
        vkCmdBindPipeline(
            cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            inputs.pipeline->pipeline);
        vkCmdBindDescriptorSets(
            cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            inputs.pipeline->pipelineLayout,
            0,
            1,
            &inputs.descriptorSet,
            0,
            nullptr);

        const uint32_t groupCountX = (inputs.width + 7u) / 8u;
        const uint32_t groupCountY = (inputs.height + 7u) / 8u;
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);
    }
}
