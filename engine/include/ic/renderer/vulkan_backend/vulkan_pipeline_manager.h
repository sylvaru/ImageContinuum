#pragma once

#include <cstddef>
#include <vector>
#include <deque>
#include <unordered_map>

#include <vulkan/vulkan.h>

#include "ic/renderer/render_pipeline.h"

namespace ic
{
    class PipelineLibrary;
    struct VulkanGraphicsPipeline
    {
        GraphicsPipelineHandle handle = {};
        GraphicsPipelineDesc desc = {};

        VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;

        explicit operator bool() const
        {
            return descriptorSetLayout != VK_NULL_HANDLE &&
                pipelineLayout != VK_NULL_HANDLE &&
                pipeline != VK_NULL_HANDLE;
        }
    };

    struct VulkanComputePipeline
    {
        ComputePipelineHandle handle = {};
        ComputePipelineDesc desc = {};

        VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;

        explicit operator bool() const
        {
            return pipelineLayout != VK_NULL_HANDLE &&
                pipeline != VK_NULL_HANDLE;
        }
    };

    class VulkanPipelineManager final
    {
    public:
        void init(VkDevice device);
        void shutdown();

        GraphicsPipelineHandle requestGraphicsPipeline(
            const GraphicsPipelineDesc& desc);
        ComputePipelineHandle requestComputePipeline(
            const ComputePipelineDesc& desc);
        GraphicsPipelineHandle resolveGraphicsPipeline(
            const PipelineLibrary& library,
            PipelineId id,
            TextureFormat swapchainFormat);
        ComputePipelineHandle resolveComputePipeline(
            const PipelineLibrary& library,
            PipelineId id);

        VulkanGraphicsPipeline* graphicsPipeline(GraphicsPipelineHandle handle);
        const VulkanGraphicsPipeline* graphicsPipeline(GraphicsPipelineHandle handle) const;
        VulkanComputePipeline* computePipeline(ComputePipelineHandle handle);
        const VulkanComputePipeline* computePipeline(ComputePipelineHandle handle) const;

    private:
        VulkanGraphicsPipeline createGraphicsPipeline(
            GraphicsPipelineHandle handle,
            const GraphicsPipelineDesc& desc) const;

        VulkanComputePipeline createComputePipeline(
            ComputePipelineHandle handle,
            const ComputePipelineDesc& desc) const;

        VkDescriptorSetLayout createDescriptorSetLayout(
            PipelineBindingLayoutKind layout) const;

        std::vector<VkVertexInputAttributeDescription> createInputAttributes(
            VertexLayoutKind layout) const;

        VkShaderModule createShaderModule(
            const std::vector<std::byte>& bytecode) const;

        VkPrimitiveTopology toVkTopology(PrimitiveTopologyKind topology) const;
        VkCullModeFlags toVkCullMode(CullMode mode) const;
        VkCompareOp toVkCompareOp(CompareOp compare) const;
        VkFormat toVkFormat(TextureFormat format) const;

        VkDevice m_device = VK_NULL_HANDLE;
        // Pipeline pointers are handed to pass recording code. deque keeps
        // element addresses stable as lazily resolved pipelines are appended.
        std::deque<VulkanGraphicsPipeline> m_graphicsPipelines;
        std::deque<VulkanComputePipeline> m_computePipelines;
        std::unordered_map<PipelineId, GraphicsPipelineHandle, PipelineIdHash>
            m_graphicsById;
        std::unordered_map<PipelineId, ComputePipelineHandle, PipelineIdHash>
            m_computeById;
    };
}
