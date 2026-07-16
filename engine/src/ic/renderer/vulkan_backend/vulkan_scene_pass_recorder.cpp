#include "ic/renderer/vulkan_backend/vulkan_scene_pass_recorder.h"

#include "ic/scene/scene_render_view.h"
#include "ic/renderer/renderer_common/renderer_util.h"

#include <cstring>

namespace ic
{
    void recordClusteredForwardCompute(
        const VulkanPassContext& ctx,
        const VulkanComputePipeline& pipeline,
        VkDescriptorSet descriptorSet)
    {
        vkCmdBindDescriptorSets(
            ctx.cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.pipelineLayout,
            0,
            1,
            &descriptorSet,
            0,
            nullptr);
    }

    void recordForwardScene(
        const VulkanPassContext& ctx,
        const VulkanForwardSceneInputs& inputs)
    {
        VkCommandBuffer cmd = ctx.cmd;
        const VkExtent2D extent = ctx.surfaceExtent;

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(extent.width);
        viewport.height = static_cast<float>(extent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = extent;

        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        vkCmdBindPipeline(
            cmd,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            inputs.pipeline->pipeline);

        vkCmdBindDescriptorSets(
            cmd,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            inputs.pipeline->pipelineLayout,
            0,
            1,
            &inputs.descriptorSet,
            0,
            nullptr);

        // Shared by the depth prepass and the forward pass: both pipelines route
        // scene geometry through this same recorder (see vulkan_pass_recorders.h
        // for why there is no separate depth-only path).
        recordSceneGeometryDraws(
            cmd,
            inputs.pipeline->pipelineLayout,
            inputs.draws,
            inputs.geometryBins,
            inputs.useGpuDriven,
            inputs.indirectStream,
            inputs.resolveNativeModel);
    }

    void recordSkybox(
        const VulkanPassContext& ctx,
        const VulkanSkyboxInputs& inputs)
    {
        const SceneRenderView& scene = *ctx.scene;
        const VkExtent2D extent = ctx.surfaceExtent;

        SkyboxConstants constants{};
        fillSkyboxConstants(
            scene.camera,
            extent.width,
            extent.height,
            scene.environment.settings,
            constants);

        std::memcpy(
            inputs.constants->mapped,
            &constants,
            sizeof(constants));
        ctx.allocator->flush(*inputs.constants, 0, sizeof(constants));

        VkCommandBuffer cmd = ctx.cmd;
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(extent.width);
        viewport.height = static_cast<float>(extent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = extent;

        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdBindPipeline(
            cmd,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            inputs.pipeline->pipeline);
        vkCmdBindDescriptorSets(
            cmd,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            inputs.pipeline->pipelineLayout,
            0,
            1,
            &inputs.descriptorSet,
            0,
            nullptr);
        vkCmdDraw(cmd, 3, 1, 0, 0);
    }
}
