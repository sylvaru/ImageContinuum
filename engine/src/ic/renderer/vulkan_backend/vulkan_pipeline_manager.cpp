#include "ic/renderer/vulkan_backend/vulkan_pipeline_manager.h"

#include <fstream>
#include <stdexcept>

#include "ic/core/asset_manager.h"
#include "ic/renderer/renderer_gpu_assets.h"
#include "ic/renderer/renderer_common/renderer_util.h"
#include "ic/renderer/pipeline_library.h"
#include "ic/util/util.h"
namespace
{
    void throwIfFailed(VkResult result, const char* message)
    {
        if (result != VK_SUCCESS)
        {
            throw std::runtime_error(message);
        }
    }
}

namespace ic
{
    void VulkanPipelineManager::init(VkDevice device)
    {
        m_device = device;
        if (m_device == VK_NULL_HANDLE)
        {
            throw std::runtime_error(
                "VulkanPipelineManager requires a valid device.");
        }
    }

    void VulkanPipelineManager::shutdown()
    {
        for (VulkanGraphicsPipeline& pipeline : m_graphicsPipelines)
        {
            if (pipeline.pipeline != VK_NULL_HANDLE)
            {
                vkDestroyPipeline(m_device, pipeline.pipeline, nullptr);
            }
            if (pipeline.pipelineLayout != VK_NULL_HANDLE)
            {
                vkDestroyPipelineLayout(
                    m_device,
                    pipeline.pipelineLayout,
                    nullptr);
            }
            if (pipeline.descriptorSetLayout != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorSetLayout(
                    m_device,
                    pipeline.descriptorSetLayout,
                    nullptr);
            }
        }
        for (VulkanComputePipeline& pipeline : m_computePipelines)
        {
            if (pipeline.pipeline != VK_NULL_HANDLE)
            {
                vkDestroyPipeline(m_device, pipeline.pipeline, nullptr);
            }
            if (pipeline.pipelineLayout != VK_NULL_HANDLE)
            {
                vkDestroyPipelineLayout(
                    m_device,
                    pipeline.pipelineLayout,
                    nullptr);
            }
            if (pipeline.descriptorSetLayout != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorSetLayout(
                    m_device,
                    pipeline.descriptorSetLayout,
                    nullptr);
            }
        }

        m_graphicsPipelines.clear();
        m_computePipelines.clear();
        m_graphicsById.clear();
        m_computeById.clear();
        m_device = VK_NULL_HANDLE;
    }

    GraphicsPipelineHandle VulkanPipelineManager::requestGraphicsPipeline(
        const GraphicsPipelineDesc& desc)
    {
        for (const VulkanGraphicsPipeline& pipeline : m_graphicsPipelines)
        {
            if (pipeline.desc.debugName == desc.debugName)
            {
                return pipeline.handle;
            }
        }

        GraphicsPipelineHandle handle{};
        handle.index = static_cast<uint32_t>(m_graphicsPipelines.size());
        handle.generation = 1;

        m_graphicsPipelines.push_back(
            createGraphicsPipeline(handle, desc));

        return handle;
    }

    ComputePipelineHandle VulkanPipelineManager::requestComputePipeline(
        const ComputePipelineDesc& desc)
    {
        for (const VulkanComputePipeline& pipeline : m_computePipelines)
        {
            if (pipeline.desc.debugName == desc.debugName)
            {
                return pipeline.handle;
            }
        }

        ComputePipelineHandle handle{};
        handle.index = static_cast<uint32_t>(m_computePipelines.size());
        handle.generation = 1;

        m_computePipelines.push_back(
            createComputePipeline(handle, desc));

        return handle;
    }

    GraphicsPipelineHandle VulkanPipelineManager::resolveGraphicsPipeline(
        const PipelineLibrary& library,
        PipelineId id,
        TextureFormat swapchainFormat)
    {
        if (!id) return {};
        if (const auto it = m_graphicsById.find(id); it != m_graphicsById.end())
            return it->second;
        const GraphicsPipelineHandle handle = requestGraphicsPipeline(
            library.resolveGraphics(
                id, RendererBackendType::Vulkan, swapchainFormat));
        m_graphicsById.emplace(id, handle);
        return handle;
    }

    ComputePipelineHandle VulkanPipelineManager::resolveComputePipeline(
        const PipelineLibrary& library,
        PipelineId id)
    {
        if (!id) return {};
        if (const auto it = m_computeById.find(id); it != m_computeById.end())
            return it->second;
        const ComputePipelineHandle handle = requestComputePipeline(
            library.resolveCompute(id, RendererBackendType::Vulkan));
        m_computeById.emplace(id, handle);
        return handle;
    }

    VulkanGraphicsPipeline* VulkanPipelineManager::graphicsPipeline(
        GraphicsPipelineHandle handle)
    {
        return const_cast<VulkanGraphicsPipeline*>(
            static_cast<const VulkanPipelineManager&>(*this).graphicsPipeline(handle));
    }

    const VulkanGraphicsPipeline* VulkanPipelineManager::graphicsPipeline(
        GraphicsPipelineHandle handle) const
    {
        if (!handle || handle.index >= m_graphicsPipelines.size())
        {
            return nullptr;
        }

        const VulkanGraphicsPipeline& pipeline =
            m_graphicsPipelines[handle.index];
        return pipeline.handle.generation == handle.generation
            ? &pipeline
            : nullptr;
    }

    VulkanComputePipeline* VulkanPipelineManager::computePipeline(
        ComputePipelineHandle handle)
    {
        return const_cast<VulkanComputePipeline*>(
            static_cast<const VulkanPipelineManager&>(*this).computePipeline(handle));
    }

    const VulkanComputePipeline* VulkanPipelineManager::computePipeline(
        ComputePipelineHandle handle) const
    {
        if (!handle || handle.index >= m_computePipelines.size())
        {
            return nullptr;
        }

        const VulkanComputePipeline& pipeline =
            m_computePipelines[handle.index];
        return pipeline.handle.generation == handle.generation
            ? &pipeline
            : nullptr;
    }

    VulkanGraphicsPipeline VulkanPipelineManager::createGraphicsPipeline(
        GraphicsPipelineHandle handle,
        const GraphicsPipelineDesc& desc) const
    {
        if (m_device == VK_NULL_HANDLE)
        {
            throw std::runtime_error(
                "VulkanPipelineManager is not initialized.");
        }

        const std::vector<std::byte> vs =
            readBinaryFile(desc.shaders.vertexShader);
        const std::vector<std::byte> ps =
            readBinaryFile(desc.shaders.pixelShader);

        VkShaderModule vertexShader = createShaderModule(vs);
        VkShaderModule pixelShader = createShaderModule(ps);

        VulkanGraphicsPipeline pipeline{};
        pipeline.handle = handle;
        pipeline.desc = desc;
        pipeline.descriptorSetLayout =
            createDescriptorSetLayout(desc.bindingLayout);

        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags =
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(DrawConstants);

        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &pipeline.descriptorSetLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushConstantRange;

        throwIfFailed(
            vkCreatePipelineLayout(
                m_device,
                &layoutInfo,
                nullptr,
                &pipeline.pipelineLayout),
            "Failed to create Vulkan pipeline layout.");

        VkPipelineShaderStageCreateInfo shaderStages[2]{};
        shaderStages[0].sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        shaderStages[0].module = vertexShader;
        shaderStages[0].pName = "VSMain";

        shaderStages[1].sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        shaderStages[1].module = pixelShader;
        shaderStages[1].pName = "PSMain";

        const std::vector<VkVertexInputAttributeDescription> attributes =
            createInputAttributes(desc.vertexLayout);
        VkVertexInputBindingDescription vertexBinding{};
        vertexBinding.binding = 0;
        vertexBinding.stride = sizeof(AssetVertex);
        vertexBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        const bool hasVertexInput =
            desc.vertexLayout == VertexLayoutKind::AssetVertex;

        VkPipelineVertexInputStateCreateInfo vertexInput{};
        vertexInput.sType =
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInput.vertexBindingDescriptionCount =
            hasVertexInput ? 1u : 0u;
        vertexInput.pVertexBindingDescriptions =
            hasVertexInput ? &vertexBinding : nullptr;
        vertexInput.vertexAttributeDescriptionCount =
            static_cast<uint32_t>(attributes.size());
        vertexInput.pVertexAttributeDescriptions = attributes.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType =
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = toVkTopology(desc.topology);

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType =
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterState{};
        rasterState.sType =
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterState.polygonMode = VK_POLYGON_MODE_FILL;
        rasterState.cullMode = toVkCullMode(desc.raster.cullMode);
        rasterState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterState.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo multisampleState{};
        multisampleState.sType =
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthState{};
        depthState.sType =
            VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthState.depthTestEnable = desc.depth.enabled ? VK_TRUE : VK_FALSE;
        depthState.depthWriteEnable = desc.depth.write ? VK_TRUE : VK_FALSE;
        depthState.depthCompareOp = toVkCompareOp(desc.depth.compare);

        VkPipelineColorBlendAttachmentState colorAttachment{};
        colorAttachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;
        colorAttachment.blendEnable =
            desc.blend.enabled ? VK_TRUE : VK_FALSE;

        VkPipelineColorBlendStateCreateInfo blendState{};
        blendState.sType =
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        blendState.attachmentCount = desc.colorAttachmentCount;
        blendState.pAttachments =
            desc.colorAttachmentCount > 0 ? &colorAttachment : nullptr;

        VkDynamicState dynamicStates[] =
        {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };

        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType =
            VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount =
            static_cast<uint32_t>(std::size(dynamicStates));
        dynamicState.pDynamicStates = dynamicStates;

        std::vector<VkFormat> colorFormats;
        colorFormats.reserve(desc.colorAttachmentCount);
        for (uint32_t i = 0; i < desc.colorAttachmentCount; ++i)
        {
            colorFormats.push_back(toVkFormat(desc.colorFormats[i]));
        }

        VkPipelineRenderingCreateInfo renderingInfo{};
        renderingInfo.sType =
            VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        renderingInfo.colorAttachmentCount =
            static_cast<uint32_t>(colorFormats.size());
        renderingInfo.pColorAttachmentFormats = colorFormats.data();
        renderingInfo.depthAttachmentFormat = toVkFormat(desc.depth.format);

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.pNext = &renderingInfo;
        pipelineInfo.stageCount = static_cast<uint32_t>(std::size(shaderStages));
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInput;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterState;
        pipelineInfo.pMultisampleState = &multisampleState;
        pipelineInfo.pDepthStencilState = &depthState;
        pipelineInfo.pColorBlendState = &blendState;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = pipeline.pipelineLayout;

        VkResult result =
            vkCreateGraphicsPipelines(
                m_device,
                VK_NULL_HANDLE,
                1,
                &pipelineInfo,
                nullptr,
                &pipeline.pipeline);

        vkDestroyShaderModule(m_device, pixelShader, nullptr);
        vkDestroyShaderModule(m_device, vertexShader, nullptr);

        throwIfFailed(result, "Failed to create Vulkan graphics pipeline.");

        return pipeline;
    }

    VulkanComputePipeline VulkanPipelineManager::createComputePipeline(
        ComputePipelineHandle handle,
        const ComputePipelineDesc& desc) const
    {
        if (m_device == VK_NULL_HANDLE)
        {
            throw std::runtime_error(
                "VulkanPipelineManager is not initialized.");
        }

        const std::vector<std::byte> cs =
            readBinaryFile(desc.shaders.computeShader);

        VkShaderModule computeShader = createShaderModule(cs);

        VulkanComputePipeline pipeline{};
        pipeline.handle = handle;
        pipeline.desc = desc;
        pipeline.descriptorSetLayout =
            createDescriptorSetLayout(desc.bindingLayout);

        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        if (pipeline.descriptorSetLayout != VK_NULL_HANDLE)
        {
            layoutInfo.setLayoutCount = 1;
            layoutInfo.pSetLayouts = &pipeline.descriptorSetLayout;
        }

        throwIfFailed(
            vkCreatePipelineLayout(
                m_device,
                &layoutInfo,
                nullptr,
                &pipeline.pipelineLayout),
            "Failed to create Vulkan compute pipeline layout.");

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = computeShader;
        stage.pName = "CSMain";

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = stage;
        pipelineInfo.layout = pipeline.pipelineLayout;

        VkResult result =
            vkCreateComputePipelines(
                m_device,
                VK_NULL_HANDLE,
                1,
                &pipelineInfo,
                nullptr,
                &pipeline.pipeline);

        vkDestroyShaderModule(m_device, computeShader, nullptr);

        throwIfFailed(result, "Failed to create Vulkan compute pipeline.");

        return pipeline;
    }

    VkDescriptorSetLayout VulkanPipelineManager::createDescriptorSetLayout(
        PipelineBindingLayoutKind layout) const
    {
        if (layout == PipelineBindingLayoutKind::Empty)
        {
            return VK_NULL_HANDLE;
        }

        if (layout == PipelineBindingLayoutKind::ComputeStorageBuffer)
        {
            VkDescriptorSetLayoutBinding binding{};
            binding.binding = 0;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binding.descriptorCount = 1;
            binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            VkDescriptorSetLayoutCreateInfo layoutInfo{};
            layoutInfo.sType =
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount = 1;
            layoutInfo.pBindings = &binding;

            VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
            throwIfFailed(
                vkCreateDescriptorSetLayout(
                    m_device,
                    &layoutInfo,
                    nullptr,
                    &setLayout),
                "Failed to create Vulkan compute descriptor set layout.");

            return setLayout;
        }

        if (layout == PipelineBindingLayoutKind::PathTrace ||
            layout == PipelineBindingLayoutKind::PathTraceTonemap)
        {
            VkDescriptorSetLayoutBinding bindings[10]{};

            bindings[0].binding = 0;
            bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            bindings[0].descriptorCount = 1;
            bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            bindings[1].binding = 1;
            bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[1].descriptorCount = 1;
            bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            const uint32_t bindingCount =
                layout == PipelineBindingLayoutKind::PathTrace
                    ? 10u
                    : 3u;

            if (layout == PipelineBindingLayoutKind::PathTrace)
            {
                for (uint32_t i = 2; i < 6u; ++i)
                {
                    bindings[i].binding = i;
                    bindings[i].descriptorType =
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    bindings[i].descriptorCount = 1;
                    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                }

                bindings[6].binding = 6;
                bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                bindings[6].descriptorCount = 1;
                bindings[6].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

                bindings[7].binding = 7;
                bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                bindings[7].descriptorCount = 1;
                bindings[7].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

                bindings[8].binding = MaxBindlessTextures + 2;
                bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                bindings[8].descriptorCount = MaxBindlessTextures;
                bindings[8].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

                bindings[9].binding = MaxBindlessSamplers;
                bindings[9].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                bindings[9].descriptorCount = MaxBindlessSamplers;
                bindings[9].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            }
            else
            {
                bindings[2].binding = 2;
                bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                bindings[2].descriptorCount = 1;
                bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            }

            VkDescriptorSetLayoutCreateInfo layoutInfo{};
            layoutInfo.sType =
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount = bindingCount;
            layoutInfo.pBindings = bindings;

            VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
            throwIfFailed(
                vkCreateDescriptorSetLayout(
                    m_device,
                    &layoutInfo,
                    nullptr,
                    &setLayout),
                "Failed to create Vulkan path tracing descriptor set layout.");

            return setLayout;
        }

        if (layout == PipelineBindingLayoutKind::EnvironmentConvert)
        {
            VkDescriptorSetLayoutBinding bindings[3]{};
            bindings[0].binding = 0;
            bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            bindings[0].descriptorCount = 1;
            bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            bindings[1].binding = 1;
            bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[1].descriptorCount = 1;
            bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            bindings[2].binding = 2;
            bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            bindings[2].descriptorCount = 1;
            bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            VkDescriptorSetLayoutCreateInfo layoutInfo{};
            layoutInfo.sType =
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount =
                static_cast<uint32_t>(std::size(bindings));
            layoutInfo.pBindings = bindings;

            VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
            throwIfFailed(
                vkCreateDescriptorSetLayout(
                    m_device,
                    &layoutInfo,
                    nullptr,
                    &setLayout),
                "Failed to create Vulkan environment descriptor set layout.");

            return setLayout;
        }

        if (layout == PipelineBindingLayoutKind::HiZDepthPyramid)
        {
            VkDescriptorSetLayoutBinding bindings[3]{};
            bindings[0] = { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[1] = { 20, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
                VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[2] = { 21, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

            VkDescriptorSetLayoutCreateInfo layoutInfo{};
            layoutInfo.sType =
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount =
                static_cast<uint32_t>(std::size(bindings));
            layoutInfo.pBindings = bindings;

            VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
            throwIfFailed(
                vkCreateDescriptorSetLayout(
                    m_device,
                    &layoutInfo,
                    nullptr,
                    &setLayout),
                "Failed to create Vulkan Hi-Z descriptor set layout.");

            return setLayout;
        }

        if (layout == PipelineBindingLayoutKind::GpuFrustumCull)
        {
            VkDescriptorSetLayoutBinding bindings[8]{};
            bindings[0] = { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[1] = { 22, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[2] = { 23, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[3] = { 24, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[4] = { 25, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[5] = { 26, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[6] = { 27, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[7] = { 28, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

            VkDescriptorSetLayoutCreateInfo layoutInfo{};
            layoutInfo.sType =
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount =
                static_cast<uint32_t>(std::size(bindings));
            layoutInfo.pBindings = bindings;

            VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
            throwIfFailed(
                vkCreateDescriptorSetLayout(
                    m_device,
                    &layoutInfo,
                    nullptr,
                    &setLayout),
                "Failed to create Vulkan GPU frustum cull descriptor set layout.");

            return setLayout;
        }

        if (layout == PipelineBindingLayoutKind::Skybox)
        {
            VkDescriptorSetLayoutBinding bindings[3]{};
            bindings[0].binding = 0;
            bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            bindings[0].descriptorCount = 1;
            bindings[0].stageFlags =
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

            bindings[1].binding = 1;
            bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            bindings[1].descriptorCount = 1;
            bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

            bindings[2].binding = 100;
            bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            bindings[2].descriptorCount = 1;
            bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

            VkDescriptorSetLayoutCreateInfo layoutInfo{};
            layoutInfo.sType =
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount =
                static_cast<uint32_t>(std::size(bindings));
            layoutInfo.pBindings = bindings;

            VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
            throwIfFailed(
                vkCreateDescriptorSetLayout(
                    m_device,
                    &layoutInfo,
                    nullptr,
                    &setLayout),
                "Failed to create Vulkan skybox descriptor set layout.");

            return setLayout;
        }

        if (layout == PipelineBindingLayoutKind::ClusteredForward)
        {
            VkDescriptorSetLayoutBinding bindings[15]{};
            bindings[0] = { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
            bindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
            bindings[3] = { 3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, MaxBindlessTextures,
                VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
            bindings[4] = { 100, VK_DESCRIPTOR_TYPE_SAMPLER, MaxBindlessSamplers,
                VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
            bindings[5] = { MaxBindlessTextures + 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
                VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
            bindings[6] = { MaxBindlessTextures + 3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
                VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
            bindings[7] = { MaxBindlessTextures + 4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
                VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
            bindings[8] = { MaxBindlessSamplers, VK_DESCRIPTOR_TYPE_SAMPLER, 1,
                VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
            bindings[9] = { 10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
            bindings[10] = { 11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
            bindings[11] = { 12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
            bindings[12] = { 13, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
            bindings[13] = { 14, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[14] = { 25, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                VK_SHADER_STAGE_VERTEX_BIT, nullptr };

            VkDescriptorSetLayoutCreateInfo layoutInfo{};
            layoutInfo.sType =
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount =
                static_cast<uint32_t>(std::size(bindings));
            layoutInfo.pBindings = bindings;

            VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
            throwIfFailed(
                vkCreateDescriptorSetLayout(
                    m_device,
                    &layoutInfo,
                    nullptr,
                    &setLayout),
                "Failed to create Vulkan clustered forward descriptor set layout.");

            return setLayout;
        }

        if (layout != PipelineBindingLayoutKind::ForwardBindless)
        {
            throw std::runtime_error(
                "Unsupported Vulkan pipeline binding layout.");
        }

        VkDescriptorSetLayoutBinding bindings[10]{};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags =
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags =
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        bindings[2].binding = 2;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags =
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        bindings[3].binding = 3;
        bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        bindings[3].descriptorCount = MaxBindlessTextures;
        bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        bindings[4].binding = 100;
        bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        bindings[4].descriptorCount = MaxBindlessSamplers;
        bindings[4].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        bindings[5].binding = MaxBindlessTextures + 2;
        bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        bindings[5].descriptorCount = 1;
        bindings[5].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        bindings[6].binding = MaxBindlessTextures + 3;
        bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        bindings[6].descriptorCount = 1;
        bindings[6].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        bindings[7].binding = MaxBindlessTextures + 4;
        bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        bindings[7].descriptorCount = 1;
        bindings[7].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        bindings[8].binding = MaxBindlessSamplers;
        bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        bindings[8].descriptorCount = 1;
        bindings[8].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        bindings[9].binding = 25;
        bindings[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[9].descriptorCount = 1;
        bindings[9].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount =
            static_cast<uint32_t>(std::size(bindings));
        layoutInfo.pBindings = bindings;

        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
        throwIfFailed(
            vkCreateDescriptorSetLayout(
                m_device,
                &layoutInfo,
                nullptr,
                &setLayout),
            "Failed to create Vulkan descriptor set layout.");

        return setLayout;
    }

    std::vector<VkVertexInputAttributeDescription>
        VulkanPipelineManager::createInputAttributes(VertexLayoutKind layout) const
    {
        if (layout == VertexLayoutKind::Unknown)
        {
            return {};
        }

        if (layout != VertexLayoutKind::AssetVertex)
        {
            throw std::runtime_error("Unsupported Vulkan vertex layout.");
        }

        return {
            { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(AssetVertex, position) },
            { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(AssetVertex, normal) },
            { 2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(AssetVertex, tangent) },
            { 3, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(AssetVertex, uv0) },
            { 4, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(AssetVertex, uv1) },
            { 5, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(AssetVertex, color) },
        };
    }

    VkShaderModule VulkanPipelineManager::createShaderModule(
        const std::vector<std::byte>& bytecode) const
    {
        VkShaderModuleCreateInfo moduleInfo{};
        moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        moduleInfo.codeSize = bytecode.size();
        moduleInfo.pCode =
            reinterpret_cast<const uint32_t*>(bytecode.data());

        VkShaderModule module = VK_NULL_HANDLE;
        throwIfFailed(
            vkCreateShaderModule(
                m_device,
                &moduleInfo,
                nullptr,
                &module),
            "Failed to create Vulkan shader module.");
        return module;
    }

    VkPrimitiveTopology VulkanPipelineManager::toVkTopology(
        PrimitiveTopologyKind topology) const
    {
        switch (topology)
        {
        case PrimitiveTopologyKind::TriangleList:
            return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        }

        return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }

    VkCullModeFlags VulkanPipelineManager::toVkCullMode(CullMode mode) const
    {
        switch (mode)
        {
        case CullMode::None:
            return VK_CULL_MODE_NONE;
        case CullMode::Front:
            return VK_CULL_MODE_FRONT_BIT;
        case CullMode::Back:
            return VK_CULL_MODE_BACK_BIT;
        }

        return VK_CULL_MODE_BACK_BIT;
    }

    VkCompareOp VulkanPipelineManager::toVkCompareOp(CompareOp compare) const
    {
        switch (compare)
        {
        case CompareOp::Never:
            return VK_COMPARE_OP_NEVER;
        case CompareOp::Less:
            return VK_COMPARE_OP_LESS;
        case CompareOp::Equal:
            return VK_COMPARE_OP_EQUAL;
        case CompareOp::LessEqual:
            return VK_COMPARE_OP_LESS_OR_EQUAL;
        case CompareOp::Greater:
            return VK_COMPARE_OP_GREATER;
        case CompareOp::Always:
            return VK_COMPARE_OP_ALWAYS;
        }

        return VK_COMPARE_OP_LESS_OR_EQUAL;
    }

    VkFormat VulkanPipelineManager::toVkFormat(TextureFormat format) const
    {
        switch (format)
        {
        case TextureFormat::RGBA8_UNorm:
            return VK_FORMAT_R8G8B8A8_UNORM;
        case TextureFormat::RGBA8_SRGB:
            return VK_FORMAT_R8G8B8A8_SRGB;
        case TextureFormat::RGBA32_Float:
            return VK_FORMAT_R32G32B32A32_SFLOAT;
        case TextureFormat::BGRA8_UNorm:
            return VK_FORMAT_B8G8R8A8_UNORM;
        case TextureFormat::BGRA8_SRGB:
            return VK_FORMAT_B8G8R8A8_SRGB;
        case TextureFormat::D32_Float:
            return VK_FORMAT_D32_SFLOAT;
        case TextureFormat::R32_Float:
            return VK_FORMAT_R32_SFLOAT;
        case TextureFormat::Unknown:
            break;
        }

        return VK_FORMAT_UNDEFINED;
    }

}
