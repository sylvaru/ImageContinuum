#include "ic/renderer/vulkan_backend/vulkan_backend.h"
#include "ic/core/app_base.h"
#include "ic/core/asset_manager.h"
#include "ic/renderer/pipeline_library.h"
#include "ic/renderer/renderer_specification.h"
#include "ic/interface/window.h"
#include "ic/core/frame_context.h"
#include "ic/scene/scene_render_view.h"

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <spdlog/spdlog.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <algorithm>
#include <future>
#include <unordered_map>

namespace ic
{
	void VulkanBackend::initialize(
		const RendererSpecification& spec,
        const PipelineLibrary& pipelineLibrary,
		Window& window,
		uint32_t workerCount)
	{
		m_instance.init();

		m_platform.init(
			m_instance.instance(),
			window);

		m_adapter.init(
			m_instance.instance(),
			m_platform.surface());

		m_device.init(
			m_adapter,
			m_platform.surface());

        m_resourceAllocator.init(
            m_instance.instance(),
            m_adapter.device(),
            m_device.device(),
            m_device.info());

		m_swapchain.init(
			m_adapter,
			m_device,
			m_platform.surface(),
			window);

        const uint32_t framesInFlight =
            spec.framesInFlight == 0 ? 1 : spec.framesInFlight;

        const uint32_t workerSlots =
            workerCount == 0 ? 1 : workerCount;
        m_workerSlots = workerSlots;

        initFrameSync(spec);

		m_commandSystem.init(
			m_device.device(),
			m_adapter.info().queueFamilies.graphics,
			framesInFlight,
			workerSlots);

        m_descriptorSystem.init(
            m_device.device(),
            m_device.info());

        m_pipelineManager.init(m_device.device());
        m_sceneFrameResources.resize(framesInFlight);
        m_pipelineLibrary = &pipelineLibrary;

        if (spec.enableImGui)
        {
            initImGui(window);
        }
	}

	void VulkanBackend::shutdown()
	{
        if (m_device.device() != VK_NULL_HANDLE)
        {
            vkDeviceWaitIdle(m_device.device());
        }

        shutdownImGui();
        destroySceneResources();
        m_pipelineManager.shutdown();

        destroySwapchainSync();
        destroyFrameSync();

        m_descriptorSystem.shutdown();
        m_resourceAllocator.shutdown();
        m_commandSystem.shutdown();
        m_swapchain.shutdown();
        m_device.shutdown();
        m_platform.shutdown();
        m_instance.shutdown();

		spdlog::info("[VulkanBackend] Shutdown");
	}

    void VulkanBackend::execute(
        const CompiledGraphPlan& plan,
        const FrameContext& ctx,
        const SceneRenderView& scene)
    {
        const uint32_t frameSlot =
            static_cast<uint32_t>(ctx.frameIndex % m_frameSync.size());

        auto& frameSync =
            m_frameSync[frameSlot];

        // CPU/GPU sync (frame pacing)
        vkWaitForFences(
            m_device.device(),
            1,
            &frameSync.inFlightFence,
            VK_TRUE,
            UINT64_MAX);

        // Acquire swapchain image
        uint32_t imageIndex =
            m_swapchain.acquireNextImage(
                frameSync.imageAvailable);

        if (imageIndex == UINT32_MAX)
        {
            onSwapchainRecreated();
            return;
        }

        m_currentSwapchainImage = imageIndex;

        VkImage swapchainImage =
            m_swapchain.image(imageIndex);

        const VkImageLayout swapchainInitialLayout =
            imageIndex < m_swapchainImageLayouts.size()
                ? m_swapchainImageLayouts[imageIndex]
                : VK_IMAGE_LAYOUT_UNDEFINED;

        if (m_imagesInFlight[imageIndex] != VK_NULL_HANDLE)
        {
            vkWaitForFences(
                m_device.device(),
                1,
                &m_imagesInFlight[imageIndex],
                VK_TRUE,
                UINT64_MAX);
        }

        m_imagesInFlight[imageIndex] =
            frameSync.inFlightFence;

        vkResetFences(
            m_device.device(),
            1,
            &frameSync.inFlightFence);

        m_commandSystem.beginFrame(frameSlot);

        std::vector<VkCommandBuffer> commandBuffers;
        executeGraph(
            plan,
            ctx,
            scene,
            swapchainImage,
            swapchainInitialLayout,
            commandBuffers);
        recordImGui(
            ctx,
            swapchainImage,
            commandBuffers);

        VkSemaphore renderFinished =
            m_imageRenderFinished[imageIndex];

        // Submit
        submitFrame(
            commandBuffers,
            frameSync,
            renderFinished);

        if (imageIndex < m_swapchainImageLayouts.size())
        {
            m_swapchainImageLayouts[imageIndex] =
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        }

        // Present
        const bool presented =
            m_swapchain.present(
            m_device.presentQueue(),
            imageIndex,
            renderFinished);

        if (!presented)
        {
            onSwapchainRecreated();
        }
    }

    bool VulkanBackend::beginDebugGuiFrame()
    {
        if (!m_imguiEnabled || m_imguiFrameActive)
        {
            return false;
        }

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        m_imguiFrameActive = true;
        return true;
    }

    void VulkanBackend::endDebugGuiFrame()
    {
        if (!m_imguiFrameActive)
        {
            return;
        }

        ImGui::Render();
        m_imguiFrameActive = false;
    }

    void VulkanBackend::initImGui(Window& window)
    {
        if (m_imguiEnabled)
        {
            return;
        }

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();

        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

        ImGui::StyleColorsDark();

        auto* glfwWindow =
            static_cast<GLFWwindow*>(window.getNativeHandle());
        if (!ImGui_ImplGlfw_InitForVulkan(glfwWindow, true))
        {
            throw std::runtime_error(
                "Failed to initialize ImGui GLFW backend.");
        }

        const VkFormat colorFormat = m_swapchain.format();
        VkPipelineRenderingCreateInfoKHR renderingInfo{};
        renderingInfo.sType =
            VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachmentFormats = &colorFormat;

        const uint32_t imageCount =
            std::max(2u, static_cast<uint32_t>(m_swapchain.images().size()));

        ImGui_ImplVulkan_InitInfo initInfo{};
        initInfo.ApiVersion = VK_API_VERSION_1_3;
        initInfo.Instance = m_instance.instance();
        initInfo.PhysicalDevice = m_adapter.device();
        initInfo.Device = m_device.device();
        initInfo.QueueFamily = m_adapter.info().queueFamilies.graphics;
        initInfo.Queue = m_device.graphicsQueue();
        initInfo.DescriptorPoolSize = 64;
        initInfo.MinImageCount = imageCount;
        initInfo.ImageCount = imageCount;
        initInfo.UseDynamicRendering = true;
        initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        initInfo.PipelineInfoMain.PipelineRenderingCreateInfo = renderingInfo;

        if (!ImGui_ImplVulkan_Init(&initInfo))
        {
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
            throw std::runtime_error(
                "Failed to initialize ImGui Vulkan backend.");
        }

        m_imguiEnabled = true;
    }

    void VulkanBackend::shutdownImGui()
    {
        if (!m_imguiEnabled)
        {
            return;
        }

        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        m_imguiEnabled = false;
        m_imguiFrameActive = false;
    }

    void VulkanBackend::recordImGui(
        const FrameContext& ctx,
        VkImage swapchainImage,
        std::vector<VkCommandBuffer>& commandBuffers)
    {
        if (!m_imguiEnabled)
        {
            return;
        }

        ImDrawData* drawData = ImGui::GetDrawData();
        if (!drawData || drawData->TotalVtxCount == 0)
        {
            return;
        }

        auto lease =
            m_commandSystem.acquireFrameCommandBuffer(
                static_cast<uint32_t>(ctx.frameIndex % m_frameSync.size()),
                0);

        VkCommandBuffer cmd = lease.commandBuffer();

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS)
        {
            throw std::runtime_error(
                "Failed to begin Vulkan ImGui command buffer.");
        }

        transitionImage(
            cmd,
            swapchainImage,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            0,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

        VkRenderingAttachmentInfo colorAttachment{};
        colorAttachment.sType =
            VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttachment.imageView =
            m_swapchain.imageView(m_currentSwapchainImage);
        colorAttachment.imageLayout =
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingInfo renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderingInfo.renderArea = { {0, 0}, m_swapchain.extent() };
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;

        vkCmdBeginRendering(cmd, &renderingInfo);
        ImGui_ImplVulkan_RenderDrawData(drawData, cmd);
        vkCmdEndRendering(cmd);

        transitionImage(
            cmd,
            swapchainImage,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            0,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
        {
            throw std::runtime_error(
                "Failed to end Vulkan ImGui command buffer.");
        }

        commandBuffers.push_back(cmd);
    }

    void VulkanBackend::submitFrame(
        std::span<const VkCommandBuffer> commandBuffers,
        FrameSync& sync,
        VkSemaphore renderFinished)
    {
        VkPipelineStageFlags waitStage =
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &sync.imageAvailable;
        submit.pWaitDstStageMask = &waitStage;

        submit.commandBufferCount =
            static_cast<uint32_t>(commandBuffers.size());
        submit.pCommandBuffers = commandBuffers.data();

        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &renderFinished;

        VkResult result =
            vkQueueSubmit(
            m_device.graphicsQueue(),
            1,
            &submit,
            sync.inFlightFence);

        if (result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to submit Vulkan frame.");
        }
    }

    void VulkanBackend::executeGraph(
        const CompiledGraphPlan& plan,
        const FrameContext& ctx,
        const SceneRenderView& scene,
        VkImage swapchainImage,
        VkImageLayout swapchainInitialLayout,
        std::vector<VkCommandBuffer>& commandBuffers)
    {
        auto recordNode =
            [&](GraphNodeId nodeId, uint32_t workerIndex)
            {
                auto lease =
                    m_commandSystem.acquireFrameCommandBuffer(
                        static_cast<uint32_t>(ctx.frameIndex % m_frameSync.size()),
                        workerIndex);

                VkCommandBuffer cmd =
                    lease.commandBuffer();

                VkCommandBufferBeginInfo beginInfo{};
                beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                beginInfo.pInheritanceInfo = nullptr;

                if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS)
                {
                    throw std::runtime_error(
                        "Failed to begin Vulkan command buffer.");
                }

                const ExecutionNode& node =
                    plan.nodes[nodeId];

                applyBarriers(
                    cmd,
                    plan.barriers,
                    plan.resources,
                    node,
                    swapchainImage,
                    swapchainInitialLayout);

                dispatchNode(
                    plan,
                    node,
                    ctx,
                    scene,
                    cmd,
                    swapchainImage);

                if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
                {
                    throw std::runtime_error(
                        "Failed to end Vulkan command buffer.");
                }

                return cmd;
            };

        if (plan.executionLevels.empty())
        {
            for (uint32_t i = 0; i < plan.executionOrder.size(); ++i)
            {
                commandBuffers.push_back(
                    recordNode(plan.executionOrder[i], i % m_workerSlots));
            }

            return;
        }

        for (const std::pmr::vector<GraphNodeId>& level : plan.executionLevels)
        {
            std::vector<std::future<VkCommandBuffer>> futures;
            futures.reserve(level.size());

            for (uint32_t i = 0; i < level.size(); ++i)
            {
                futures.push_back(
                    std::async(
                        std::launch::async,
                        recordNode,
                        level[i],
                        i % m_workerSlots));
            }

            for (auto& future : futures)
            {
                commandBuffers.push_back(future.get());
            }
        }
    }

    void VulkanBackend::applyBarriers(
        VkCommandBuffer cmd,
        std::span<const ResourceBarrier> barriers,
        std::span<const GraphResource> resources,
        const ExecutionNode& node,
        VkImage swapchainImage,
        VkImageLayout swapchainInitialLayout)
    {
        for (const ResourceBarrier& barrier : barriers)
        {
            if (barrier.toNode != node.nodeId)
                continue;


            recordBarrier(
                cmd,
                barrier,
                resources,
                swapchainImage,
                swapchainInitialLayout);
        }
    }

    void VulkanBackend::recordBarrier(
        VkCommandBuffer cmd,
        const ResourceBarrier& barrier,
        std::span<const GraphResource> resources,
        VkImage swapchainImage,
        VkImageLayout swapchainInitialLayout)
    {

        VkImageMemoryBarrier vkBarrier{};
        vkBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;

        const GraphResource& resource =
            resources[barrier.resource];

        VkImage image = VK_NULL_HANDLE;

        bool isSwapchainImage = false;

        if (resource.ownership == ResourceOwnership::Imported)
        {
            switch (resource.imported)
            {
            case ImportedResource::Swapchain:
                image = swapchainImage;
                isSwapchainImage = true;
                break;
            }
        }
        else
        {
            // TODO
            return;
        }
        vkBarrier.image = image;

        const bool isExternalFirstUse =
            isSwapchainImage &&
            barrier.fromNode == barrier.toNode;

        vkBarrier.oldLayout = isExternalFirstUse
            ? swapchainInitialLayout
            : usageToLayout(barrier.oldUsage);

        vkBarrier.newLayout = usageToLayout(barrier.newUsage);

        vkBarrier.srcAccessMask =
            vkBarrier.oldLayout == VK_IMAGE_LAYOUT_UNDEFINED
                ? 0
                : accessMaskFor(
                    barrier.oldUsage,
                    barrier.fromAccess);

        vkBarrier.dstAccessMask =
            accessMaskFor(
                barrier.newUsage,
                barrier.toAccess);

        vkBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vkBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        vkBarrier.srcQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;

        vkBarrier.dstQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;

        vkBarrier.subresourceRange.aspectMask =
            VK_IMAGE_ASPECT_COLOR_BIT;

        vkBarrier.subresourceRange.baseMipLevel = 0;
        vkBarrier.subresourceRange.levelCount = 1;
        vkBarrier.subresourceRange.baseArrayLayer = 0;
        vkBarrier.subresourceRange.layerCount = 1;

        VkPipelineStageFlags srcStage =
            pipelineStageFor(
                barrier.oldUsage,
                barrier.fromAccess);

        if (vkBarrier.oldLayout == VK_IMAGE_LAYOUT_UNDEFINED)
        {
            srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        }

        VkPipelineStageFlags dstStage =
            pipelineStageFor(
                barrier.newUsage,
                barrier.toAccess);

        if (barrier.newUsage == ResourceUsage::Present)
        {
            dstStage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        }

        vkCmdPipelineBarrier(
            cmd,
            srcStage,
            dstStage,
            0,
            0, nullptr,
            0, nullptr,
            1, &vkBarrier);
    }

    void VulkanBackend::dispatchNode(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node,
        const FrameContext& ctx,
        const SceneRenderView& scene,
        VkCommandBuffer cmd,
        [[maybe_unused]] VkImage swapchainImage)
    {
        switch (node.type)
        {
        case GraphNodeType::Graphics:
            executeGraphicsNode(plan, node, ctx, scene, cmd, swapchainImage);
            break;

        case GraphNodeType::Compute:
            executeComputeNode(plan, node, ctx, cmd);
            break;

        case GraphNodeType::Transfer:
            executeTransferNode(plan, node, ctx, cmd);
            break;

        case GraphNodeType::Present:
            break;
        }
    }

    void VulkanBackend::executeGraphicsNode(
        const CompiledGraphPlan& plan,
        [[maybe_unused]] const ExecutionNode& node,
        const FrameContext& ctx,
        const SceneRenderView& scene,
        VkCommandBuffer cmd,
        [[maybe_unused]] VkImage swapchainImage)
    {
        ensureDepthTarget();

        const GraphicsPipelineHandle pipelineHandle =
            pipelineForNode(plan, node);
        VulkanGraphicsPipeline* pipeline =
            m_pipelineManager.graphicsPipeline(pipelineHandle);
        if (!pipeline)
        {
            return;
        }

        const bool hasColorTarget =
            pipeline->desc.colorAttachmentCount > 0;

        VkClearValue clear{};
        clear.color = { { 0.02f, 0.02f, 0.025f, 1.0f } };

        VkImageMemoryBarrier depthBarrier{};
        depthBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        depthBarrier.oldLayout = m_depthLayout;
        depthBarrier.newLayout =
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthBarrier.srcAccessMask = 0;
        depthBarrier.dstAccessMask =
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        depthBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        depthBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        depthBarrier.image = m_depthTexture.image;
        depthBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        depthBarrier.subresourceRange.baseMipLevel = 0;
        depthBarrier.subresourceRange.levelCount = 1;
        depthBarrier.subresourceRange.baseArrayLayer = 0;
        depthBarrier.subresourceRange.layerCount = 1;

        if (m_depthLayout !=
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        {
            vkCmdPipelineBarrier(
                cmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                0,
                0, nullptr,
                0, nullptr,
                1, &depthBarrier);
            m_depthLayout =
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        }

        VkRenderingAttachmentInfo colorAttachment{};
        colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttachment.imageView = m_swapchain.imageView(m_currentSwapchainImage);
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.clearValue = clear;

        VkClearValue depthClear{};
        depthClear.depthStencil = { 1.0f, 0 };

        VkRenderingAttachmentInfo depthAttachment{};
        depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depthAttachment.imageView = m_depthImageView;
        depthAttachment.imageLayout =
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthAttachment.loadOp =
            hasColorTarget
                ? VK_ATTACHMENT_LOAD_OP_LOAD
                : VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depthAttachment.clearValue = depthClear;

        VkRenderingInfo renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderingInfo.renderArea = { {0, 0}, m_swapchain.extent() };
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = hasColorTarget ? 1u : 0u;
        renderingInfo.pColorAttachments =
            hasColorTarget ? &colorAttachment : nullptr;
        renderingInfo.pDepthAttachment = &depthAttachment;

        vkCmdBeginRendering(cmd, &renderingInfo);

        std::vector<DrawItem> draws;
        if (prepareSceneResources(ctx, scene, pipelineHandle, draws) &&
            !draws.empty())
        {
                const VkExtent2D extent = m_swapchain.extent();

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

                const uint32_t frameSlot =
                    static_cast<uint32_t>(
                        ctx.frameIndex % m_sceneFrameResources.size());
                FrameSceneResources& frameResources =
                    m_sceneFrameResources[frameSlot];

                vkCmdBindPipeline(
                    cmd,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    pipeline->pipeline);

                vkCmdBindDescriptorSets(
                    cmd,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    pipeline->pipelineLayout,
                    0,
                    1,
                    &frameResources.descriptorSet,
                    0,
                    nullptr);

                UploadedModel* boundModel = nullptr;
                for (const DrawItem& draw : draws)
                {
                    if (!draw.model ||
                        draw.meshIndex >= draw.model->meshes.size())
                    {
                        continue;
                    }

                    const GpuMesh& mesh =
                        draw.model->meshes[draw.meshIndex];
                    if (mesh.indexCount == 0)
                    {
                        continue;
                    }

                    if (boundModel != draw.model)
                    {
                        VkDeviceSize offset = 0;
                        vkCmdBindVertexBuffers(
                            cmd,
                            0,
                            1,
                            &draw.model->vertexBuffer.buffer,
                            &offset);
                        vkCmdBindIndexBuffer(
                            cmd,
                            draw.model->indexBuffer.buffer,
                            0,
                            VK_INDEX_TYPE_UINT32);
                        boundModel = draw.model;
                    }

                    DrawConstants constants{};
                    constants.objectIndex = draw.objectIndex;
                    constants.meshIndex = draw.meshIndex;
                    constants.materialIndex = draw.materialIndex;

                    vkCmdPushConstants(
                        cmd,
                        pipeline->pipelineLayout,
                        VK_SHADER_STAGE_VERTEX_BIT |
                            VK_SHADER_STAGE_FRAGMENT_BIT,
                        0,
                        sizeof(constants),
                        &constants);

                    vkCmdDrawIndexed(
                        cmd,
                        mesh.indexCount,
                        1,
                        mesh.firstIndex,
                        0,
                        0);
                }
        }

        vkCmdEndRendering(cmd);

    }

    void VulkanBackend::executeComputeNode(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node,
        [[maybe_unused]] const FrameContext& ctx,
        [[maybe_unused]] VkCommandBuffer cmd)
    {
        if (node.payloadIndex >= plan.payloads.size())
        {
            return;
        }

        const ComputePassData* pass =
            std::get_if<ComputePassData>(&plan.payloads[node.payloadIndex]);
        if (!pass || !pass->pipeline)
        {
            return;
        }

        VulkanComputePipeline* pipeline =
            m_pipelineManager.computePipeline(
                computePipelineForNode(plan, node));
        if (!pipeline)
        {
            return;
        }

        vkCmdBindPipeline(
            cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline->pipeline);

        if (pipeline->desc.bindingLayout ==
            PipelineBindingLayoutKind::ComputeStorageBuffer)
        {
            ensureComputeTestResources(*pipeline);
            vkCmdBindDescriptorSets(
                cmd,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline->pipelineLayout,
                0,
                1,
                &m_computeTestDescriptorSet,
                0,
                nullptr);
        }

        vkCmdDispatch(
            cmd,
            pass->groupCountX,
            pass->groupCountY,
            pass->groupCountZ);

        if (pipeline->desc.bindingLayout ==
            PipelineBindingLayoutKind::ComputeStorageBuffer)
        {
            VkBufferMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask =
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = m_computeTestBuffer.buffer;
            barrier.offset = 0;
            barrier.size = m_computeTestBuffer.size;

            vkCmdPipelineBarrier(
                cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr,
                1, &barrier,
                0, nullptr);
        }
    }

    void VulkanBackend::executeTransferNode(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node,
        [[maybe_unused]] const FrameContext& ctx,
        [[maybe_unused]] VkCommandBuffer cmd)
    {
        if (node.payloadIndex >= plan.payloads.size())
        {
            return;
        }

        const TransferPassData* pass =
            std::get_if<TransferPassData>(&plan.payloads[node.payloadIndex]);
        if (!pass)
        {
            return;
        }
    }

    GraphicsPipelineHandle VulkanBackend::pipelineForNode(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node)
    {
        if (!m_pipelineLibrary ||
            node.payloadIndex >= plan.payloads.size())
        {
            return {};
        }

        const GraphicsPassData* pass =
            std::get_if<GraphicsPassData>(&plan.payloads[node.payloadIndex]);
        if (!pass || !pass->pipeline)
        {
            return {};
        }

        auto it = m_pipelineHandles.find(pass->pipeline);
        if (it != m_pipelineHandles.end())
        {
            return it->second;
        }

        GraphicsPipelineDesc desc =
            m_pipelineLibrary->resolveGraphics(
                pass->pipeline,
                RendererBackendType::Vulkan,
                swapchainTextureFormat());

        GraphicsPipelineHandle handle =
            m_pipelineManager.requestGraphicsPipeline(desc);
        m_pipelineHandles.emplace(pass->pipeline, handle);
        return handle;
    }

    ComputePipelineHandle VulkanBackend::computePipelineForNode(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node)
    {
        if (!m_pipelineLibrary ||
            node.payloadIndex >= plan.payloads.size())
        {
            return {};
        }

        const ComputePassData* pass =
            std::get_if<ComputePassData>(&plan.payloads[node.payloadIndex]);
        if (!pass || !pass->pipeline)
        {
            return {};
        }

        auto it = m_computePipelineHandles.find(pass->pipeline);
        if (it != m_computePipelineHandles.end())
        {
            return it->second;
        }

        ComputePipelineDesc desc =
            m_pipelineLibrary->resolveCompute(
                pass->pipeline,
                RendererBackendType::Vulkan);

        ComputePipelineHandle handle =
            m_pipelineManager.requestComputePipeline(desc);
        m_computePipelineHandles.emplace(pass->pipeline, handle);
        return handle;
    }

    void VulkanBackend::destroySceneResources()
    {
        destroyDepthTarget();

        if (m_computeTestDescriptorPool != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(
                m_device.device(),
                m_computeTestDescriptorPool,
                nullptr);
            m_computeTestDescriptorPool = VK_NULL_HANDLE;
            m_computeTestDescriptorSet = VK_NULL_HANDLE;
        }
        m_resourceAllocator.destroyBuffer(m_computeTestBuffer);

        for (auto& [handle, model] : m_uploadedModels)
        {
            m_resourceAllocator.destroyBuffer(model.vertexBuffer);
            m_resourceAllocator.destroyBuffer(model.indexBuffer);
        }
        m_uploadedModels.clear();

        for (FrameSceneResources& resources : m_sceneFrameResources)
        {
            m_resourceAllocator.destroyBuffer(resources.frameConstants);
            m_resourceAllocator.destroyBuffer(resources.objects);
            m_resourceAllocator.destroyBuffer(resources.materials);

            if (resources.descriptorPool != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorPool(
                    m_device.device(),
                    resources.descriptorPool,
                    nullptr);
            }

            resources = {};
        }
        m_sceneFrameResources.clear();

        m_pipelineHandles.clear();
        m_computePipelineHandles.clear();
    }

    void VulkanBackend::ensureComputeTestResources(
        const VulkanComputePipeline& pipeline)
    {
        if (m_computeTestBuffer &&
            m_computeTestDescriptorSet != VK_NULL_HANDLE)
        {
            return;
        }

        if (!m_computeTestBuffer)
        {
            m_computeTestBuffer =
                m_resourceAllocator.createBuffer({
                    .size = 64 * 64 * sizeof(uint32_t),
                    .usage = BufferUsageFlags::Storage,
                    .memoryUsage = ResourceMemoryUsage::GpuOnly,
                    .debugName = "Vulkan compute binding test buffer"
                });
        }

        if (m_computeTestDescriptorPool == VK_NULL_HANDLE)
        {
            VkDescriptorPoolSize poolSize{};
            poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            poolSize.descriptorCount = 1;

            VkDescriptorPoolCreateInfo poolInfo{};
            poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.maxSets = 1;
            poolInfo.poolSizeCount = 1;
            poolInfo.pPoolSizes = &poolSize;

            if (vkCreateDescriptorPool(
                    m_device.device(),
                    &poolInfo,
                    nullptr,
                    &m_computeTestDescriptorPool) != VK_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to create Vulkan compute test descriptor pool.");
            }
        }

        VkDescriptorSetAllocateInfo allocateInfo{};
        allocateInfo.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocateInfo.descriptorPool = m_computeTestDescriptorPool;
        allocateInfo.descriptorSetCount = 1;
        allocateInfo.pSetLayouts = &pipeline.descriptorSetLayout;

        if (vkAllocateDescriptorSets(
                m_device.device(),
                &allocateInfo,
                &m_computeTestDescriptorSet) != VK_SUCCESS)
        {
            throw std::runtime_error(
                "Failed to allocate Vulkan compute test descriptor set.");
        }

        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = m_computeTestBuffer.buffer;
        bufferInfo.offset = 0;
        bufferInfo.range = m_computeTestBuffer.size;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = m_computeTestDescriptorSet;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(
            m_device.device(),
            1,
            &write,
            0,
            nullptr);
    }

    void VulkanBackend::ensureDepthTarget()
    {
        const VkExtent2D extent = m_swapchain.extent();
        if (m_depthTexture &&
            m_depthWidth == extent.width &&
            m_depthHeight == extent.height)
        {
            return;
        }

        destroyDepthTarget();

        m_depthTexture =
            m_resourceAllocator.createTexture({
                .width = extent.width,
                .height = extent.height,
                .format = TextureFormat::D32_Float,
                .usage = TextureUsageFlags::DepthAttachment,
                .memoryUsage = ResourceMemoryUsage::GpuOnly,
                .debugName = "Vulkan forward depth"
            });

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = m_depthTexture.image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_D32_SFLOAT;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(
                m_device.device(),
                &viewInfo,
                nullptr,
                &m_depthImageView) != VK_SUCCESS)
        {
            throw std::runtime_error(
                "Failed to create Vulkan depth image view.");
        }

        m_depthLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        m_depthWidth = extent.width;
        m_depthHeight = extent.height;
    }

    void VulkanBackend::destroyDepthTarget()
    {
        if (m_depthImageView != VK_NULL_HANDLE)
        {
            vkDestroyImageView(
                m_device.device(),
                m_depthImageView,
                nullptr);
            m_depthImageView = VK_NULL_HANDLE;
        }

        m_resourceAllocator.destroyTexture(m_depthTexture);
        m_depthLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        m_depthWidth = 0;
        m_depthHeight = 0;
    }

    VulkanBackend::UploadedModel* VulkanBackend::requestModel(
        AssetHandle handle,
        const AssetManager& assets)
    {
        if (!handle)
        {
            return nullptr;
        }

        auto [it, inserted] =
            m_uploadedModels.try_emplace(handle);
        UploadedModel& uploaded = it->second;
        if (uploaded.uploaded)
        {
            return &uploaded;
        }

        const ModelAsset* model = assets.model(handle);
        if (!model ||
            model->vertices.empty() ||
            model->indices.empty())
        {
            return nullptr;
        }

        uploaded.vertexBuffer =
            m_resourceAllocator.createBuffer({
                .size = model->vertices.size() * sizeof(AssetVertex),
                .usage = BufferUsageFlags::Vertex,
                .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                .mappedAtCreation = true,
                .debugName = "Cornell vertex buffer"
            });

        uploaded.indexBuffer =
            m_resourceAllocator.createBuffer({
                .size = model->indices.size() * sizeof(uint32_t),
                .usage = BufferUsageFlags::Index,
                .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                .mappedAtCreation = true,
                .debugName = "Cornell index buffer"
            });

        std::memcpy(
            uploaded.vertexBuffer.mapped,
            model->vertices.data(),
            uploaded.vertexBuffer.size);
        std::memcpy(
            uploaded.indexBuffer.mapped,
            model->indices.data(),
            uploaded.indexBuffer.size);

        m_resourceAllocator.flush(
            uploaded.vertexBuffer,
            0,
            uploaded.vertexBuffer.size);
        m_resourceAllocator.flush(
            uploaded.indexBuffer,
            0,
            uploaded.indexBuffer.size);

        uploaded.materials.reserve(
            std::max<size_t>(1, model->materials.size()));
        for (const MaterialAsset& src : model->materials)
        {
            GpuMaterialData dst{};
            dst.baseColorFactor = src.baseColorFactor;
            dst.metallicFactor = src.metallicFactor;
            dst.roughnessFactor = src.roughnessFactor;
            dst.alphaCutoff = src.alphaCutoff;
            dst.flags =
                (src.doubleSided ? 1u : 0u) |
                (src.alphaBlend ? 2u : 0u) |
                (src.alphaMask ? 4u : 0u);
            uploaded.materials.push_back(dst);
        }
        if (uploaded.materials.empty())
        {
            uploaded.materials.push_back(GpuMaterialData{});
        }

        std::vector<glm::mat4> meshNodeTransforms(
            model->meshes.size(),
            glm::mat4(1.0f));

        for (const NodeAsset& node : model->nodes)
        {
            if (node.meshIndex >= 0 &&
                static_cast<size_t>(node.meshIndex) <
                    meshNodeTransforms.size())
            {
                meshNodeTransforms[static_cast<size_t>(node.meshIndex)] =
                    node.worldMatrix;
            }
        }

        for (size_t meshIndex = 0;
             meshIndex < model->meshes.size();
             ++meshIndex)
        {
            const MeshAsset& mesh = model->meshes[meshIndex];
            for (const MeshPrimitiveAsset& primitive : mesh.primitives)
            {
                GpuMesh gpuMesh{};
                gpuMesh.materialIndex =
                    primitive.materialIndex != UINT32_MAX
                        ? primitive.materialIndex
                        : 0u;
                gpuMesh.indexCount = primitive.indexCount;
                gpuMesh.firstIndex = primitive.firstIndex;
                gpuMesh.vertexOffset = primitive.firstVertex;
                uploaded.meshes.push_back(gpuMesh);
                uploaded.meshTransforms.push_back(
                    meshNodeTransforms[meshIndex]);
            }
        }

        uploaded.uploaded = true;
        return &uploaded;
    }

    bool VulkanBackend::prepareSceneResources(
        const FrameContext& ctx,
        const SceneRenderView& scene,
        GraphicsPipelineHandle pipelineHandle,
        std::vector<DrawItem>& draws)
    {
        if (!ctx.services ||
            !ctx.services->assetManager ||
            scene.camera.valid == 0 ||
            m_sceneFrameResources.empty())
        {
            return false;
        }

        AssetManager& assets = *ctx.services->assetManager;
        std::vector<GpuObjectData> objects;
        std::vector<GpuMaterialData> materials;
        std::unordered_map<AssetHandle, uint32_t, AssetHandleHash> materialOffsets;

        objects.reserve(scene.models.size());

        for (const SceneModelRenderItem& item : scene.models)
        {
            UploadedModel* model = requestModel(item.model, assets);
            if (!model)
            {
                continue;
            }

            uint32_t materialOffset = 0;
            if (auto it = materialOffsets.find(item.model);
                it != materialOffsets.end())
            {
                materialOffset = it->second;
            }
            else
            {
                materialOffset = static_cast<uint32_t>(materials.size());
                materialOffsets.emplace(item.model, materialOffset);
                materials.insert(
                    materials.end(),
                    model->materials.begin(),
                    model->materials.end());
            }

            for (uint32_t meshIndex = 0;
                 meshIndex < model->meshes.size();
                 ++meshIndex)
            {
                const GpuMesh& mesh = model->meshes[meshIndex];
                const glm::mat4 meshWorld =
                    meshIndex < model->meshTransforms.size()
                        ? item.world * model->meshTransforms[meshIndex]
                        : item.world;

                const uint32_t objectIndex =
                    static_cast<uint32_t>(objects.size());

                GpuObjectData object{};
                object.world = meshWorld;
                object.inverseTransposeWorld =
                    glm::inverseTranspose(meshWorld);
                objects.push_back(object);

                DrawItem draw{};
                draw.model = model;
                draw.objectIndex = objectIndex;
                draw.meshIndex = meshIndex;
                draw.materialIndex = materialOffset + mesh.materialIndex;
                draws.push_back(draw);
            }
        }

        if (draws.empty())
        {
            return false;
        }

        if (materials.empty())
        {
            materials.push_back(GpuMaterialData{});
        }

        const uint32_t frameSlot =
            static_cast<uint32_t>(
                ctx.frameIndex % m_sceneFrameResources.size());
        FrameSceneResources& frameResources =
            m_sceneFrameResources[frameSlot];

        if (!frameResources.frameConstants)
        {
            frameResources.frameConstants =
                m_resourceAllocator.createBuffer({
                    .size = sizeof(GpuFrameData),
                    .usage = BufferUsageFlags::Constant,
                    .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                    .mappedAtCreation = true,
                    .debugName = "Vulkan frame constants"
                });
        }

        bool descriptorsDirty = false;
        if (objects.size() > frameResources.objectCapacity)
        {
            m_resourceAllocator.destroyBuffer(frameResources.objects);
            frameResources.objects =
                m_resourceAllocator.createBuffer({
                    .size = objects.size() * sizeof(GpuObjectData),
                    .usage = BufferUsageFlags::Storage,
                    .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                    .mappedAtCreation = true,
                    .debugName = "Vulkan object data"
                });
            frameResources.objectCapacity =
                static_cast<uint32_t>(objects.size());
            descriptorsDirty = true;
        }

        if (materials.size() > frameResources.materialCapacity)
        {
            m_resourceAllocator.destroyBuffer(frameResources.materials);
            frameResources.materials =
                m_resourceAllocator.createBuffer({
                    .size = materials.size() * sizeof(GpuMaterialData),
                    .usage = BufferUsageFlags::Storage,
                    .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                    .mappedAtCreation = true,
                    .debugName = "Vulkan material data"
                });
            frameResources.materialCapacity =
                static_cast<uint32_t>(materials.size());
            descriptorsDirty = true;
        }

        if (descriptorsDirty ||
            frameResources.descriptorSet == VK_NULL_HANDLE)
        {
            updateFrameDescriptors(frameResources, pipelineHandle);
        }

        GpuFrameData frameData{};
        frameData.view = scene.camera.view;
        frameData.projection = glm::perspectiveRH_ZO(
            scene.camera.verticalFovRadians,
            scene.camera.aspectRatio,
            scene.camera.nearPlane,
            scene.camera.farPlane);
        frameData.projection[1][1] *= -1.0f;
        frameData.viewProjection = frameData.projection * frameData.view;
        frameData.cameraPosition = scene.camera.position;
        frameData.time = ctx.timeSinceStart;

        for (const SceneLightRenderItem& light : scene.lights)
        {
            if (light.type == LightType::Directional)
            {
                frameData.lightDirection = light.direction;
                frameData.lightColor = light.color;
                frameData.lightIntensity = light.intensity;
                break;
            }
        }

        std::memcpy(
            frameResources.frameConstants.mapped,
            &frameData,
            sizeof(frameData));
        std::memcpy(
            frameResources.objects.mapped,
            objects.data(),
            objects.size() * sizeof(GpuObjectData));
        std::memcpy(
            frameResources.materials.mapped,
            materials.data(),
            materials.size() * sizeof(GpuMaterialData));

        m_resourceAllocator.flush(
            frameResources.frameConstants,
            0,
            sizeof(frameData));
        m_resourceAllocator.flush(
            frameResources.objects,
            0,
            objects.size() * sizeof(GpuObjectData));
        m_resourceAllocator.flush(
            frameResources.materials,
            0,
            materials.size() * sizeof(GpuMaterialData));

        return true;
    }

    void VulkanBackend::updateFrameDescriptors(
        FrameSceneResources& resources,
        GraphicsPipelineHandle pipelineHandle)
    {
        VulkanGraphicsPipeline* pipeline =
            m_pipelineManager.graphicsPipeline(pipelineHandle);
        if (!pipeline)
        {
            return;
        }

        if (resources.descriptorPool != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(
                m_device.device(),
                resources.descriptorPool,
                nullptr);
            resources.descriptorPool = VK_NULL_HANDLE;
            resources.descriptorSet = VK_NULL_HANDLE;
        }

        VkDescriptorPoolSize poolSizes[2]{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = 1;
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[1].descriptorCount = 2;

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount =
            static_cast<uint32_t>(std::size(poolSizes));
        poolInfo.pPoolSizes = poolSizes;

        if (vkCreateDescriptorPool(
                m_device.device(),
                &poolInfo,
                nullptr,
                &resources.descriptorPool) != VK_SUCCESS)
        {
            throw std::runtime_error(
                "Failed to create Vulkan forward descriptor pool.");
        }

        VkDescriptorSetAllocateInfo allocateInfo{};
        allocateInfo.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocateInfo.descriptorPool = resources.descriptorPool;
        allocateInfo.descriptorSetCount = 1;
        allocateInfo.pSetLayouts = &pipeline->descriptorSetLayout;

        if (vkAllocateDescriptorSets(
                m_device.device(),
                &allocateInfo,
                &resources.descriptorSet) != VK_SUCCESS)
        {
            throw std::runtime_error(
                "Failed to allocate Vulkan forward descriptor set.");
        }

        VkDescriptorBufferInfo frameInfo{};
        frameInfo.buffer = resources.frameConstants.buffer;
        frameInfo.offset = 0;
        frameInfo.range = sizeof(GpuFrameData);

        VkDescriptorBufferInfo objectInfo{};
        objectInfo.buffer = resources.objects.buffer;
        objectInfo.offset = 0;
        objectInfo.range = resources.objects.size;

        VkDescriptorBufferInfo materialInfo{};
        materialInfo.buffer = resources.materials.buffer;
        materialInfo.offset = 0;
        materialInfo.range = resources.materials.size;

        VkWriteDescriptorSet writes[3]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = resources.descriptorSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].pBufferInfo = &frameInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = resources.descriptorSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &objectInfo;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = resources.descriptorSet;
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo = &materialInfo;

        vkUpdateDescriptorSets(
            m_device.device(),
            static_cast<uint32_t>(std::size(writes)),
            writes,
            0,
            nullptr);
    }


    VkImageLayout VulkanBackend::usageToLayout(ResourceUsage usage) const
    {
        switch (usage)
        {
        case ResourceUsage::SampledTexture:
            return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        case ResourceUsage::StorageTexture:
            return VK_IMAGE_LAYOUT_GENERAL;

        case ResourceUsage::ColorAttachment:
            return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        case ResourceUsage::DepthAttachment:
            return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        case ResourceUsage::Present:
            return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        case ResourceUsage::TransferDst:
            return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

        case ResourceUsage::TransferSrc:
            return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

        default:
            return VK_IMAGE_LAYOUT_GENERAL;
        }
    }

    VkAccessFlags VulkanBackend::toAccessMask(AccessType type) const
    {
        switch (type)
        {
        case AccessType::Read:
            return VK_ACCESS_SHADER_READ_BIT;

        case AccessType::Write:
            return VK_ACCESS_SHADER_WRITE_BIT;

        case AccessType::ReadWrite:
            return VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        }
        return 0;
    }

    VkAccessFlags VulkanBackend::accessMaskFor(
        ResourceUsage usage,
        AccessType access) const
    {
        if (access == AccessType::ReadWrite)
        {
            return
                accessMaskFor(usage, AccessType::Read) |
                accessMaskFor(usage, AccessType::Write);
        }

        if (access == AccessType::Read)
        {
            switch (usage)
            {
            case ResourceUsage::SampledTexture:
                return VK_ACCESS_SHADER_READ_BIT;

            case ResourceUsage::ColorAttachment:
                return VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;

            case ResourceUsage::DepthAttachment:
                return VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;

            case ResourceUsage::VertexBuffer:
                return VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;

            case ResourceUsage::IndexBuffer:
                return VK_ACCESS_INDEX_READ_BIT;

            case ResourceUsage::ConstantBuffer:
                return VK_ACCESS_UNIFORM_READ_BIT;

            case ResourceUsage::StorageBuffer:
            case ResourceUsage::StorageTexture:
                return VK_ACCESS_SHADER_READ_BIT;

            case ResourceUsage::TransferSrc:
                return VK_ACCESS_TRANSFER_READ_BIT;

            case ResourceUsage::Present:
                return 0;

            default:
                return VK_ACCESS_MEMORY_READ_BIT;
            }
        }

        switch (usage)
        {
        case ResourceUsage::ColorAttachment:
            return VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        case ResourceUsage::DepthAttachment:
            return VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        case ResourceUsage::StorageBuffer:
        case ResourceUsage::StorageTexture:
            return VK_ACCESS_SHADER_WRITE_BIT;

        case ResourceUsage::TransferDst:
            return VK_ACCESS_TRANSFER_WRITE_BIT;

        case ResourceUsage::Present:
            return 0;

        default:
            return VK_ACCESS_MEMORY_WRITE_BIT;
        }
    }

    VkPipelineStageFlags VulkanBackend::pipelineStageFor(
        ResourceUsage usage,
        AccessType access) const
    {
        switch (usage)
        {
        case ResourceUsage::ColorAttachment:
            return VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

        case ResourceUsage::DepthAttachment:
            return
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;

        case ResourceUsage::SampledTexture:
            return VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

        case ResourceUsage::StorageTexture:
        case ResourceUsage::StorageBuffer:
            return VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

        case ResourceUsage::VertexBuffer:
        case ResourceUsage::IndexBuffer:
            return VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;

        case ResourceUsage::ConstantBuffer:
            return
                VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

        case ResourceUsage::TransferDst:
        case ResourceUsage::TransferSrc:
            return VK_PIPELINE_STAGE_TRANSFER_BIT;

        case ResourceUsage::Present:
            return access == AccessType::Read
                ? VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

        default:
            return VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        }
    }

    VkImageLayout VulkanBackend::getOrInitImageLayout(VkImage image)
    {
        auto it = m_imageStates.find(image);
        if (it != m_imageStates.end())
            return it->second.layout;

        m_imageStates[image] = {
            .layout = VK_IMAGE_LAYOUT_UNDEFINED,
            .access = AccessType::Read
        };

        return VK_IMAGE_LAYOUT_UNDEFINED;
    }

    void VulkanBackend::transitionImage(
        VkCommandBuffer cmd,
        VkImage image,
        VkImageLayout oldLayout,
        VkImageLayout newLayout,
        VkAccessFlags srcAccess,
        VkAccessFlags dstAccess,
        VkPipelineStageFlags srcStage,
        VkPipelineStageFlags dstStage)
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;

        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;

        barrier.srcAccessMask = srcAccess;
        barrier.dstAccessMask = dstAccess;

        barrier.image = image;

        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;

        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        vkCmdPipelineBarrier(
            cmd,
            srcStage,
            dstStage,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier);

        m_imageStates[image].layout = newLayout;
        m_imageStates[image].access = (dstAccess != 0)
            ? AccessType::Write
            : AccessType::Read;
    }

    void VulkanBackend::initFrameSync(const RendererSpecification& spec)
    {
        const uint32_t framesInFlight =
            spec.framesInFlight == 0 ? 1 : spec.framesInFlight;

        m_frameSync.resize(framesInFlight);

        for (auto& fs : m_frameSync)
        {
            VkSemaphoreCreateInfo semInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };

            if (vkCreateSemaphore(
                m_device.device(),
                &semInfo,
                nullptr,
                &fs.imageAvailable) != VK_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to create Vulkan image-available semaphore.");
            }

            VkFenceCreateInfo fenceInfo{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
            fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

            if (vkCreateFence(
                m_device.device(),
                &fenceInfo,
                nullptr,
                &fs.inFlightFence) != VK_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to create Vulkan frame fence.");
            }
        }

        initSwapchainSync();
    }

    void VulkanBackend::initSwapchainSync()
    {
        destroySwapchainSync();

        const size_t imageCount =
            m_swapchain.images().size();

        m_imageRenderFinished.resize(imageCount, VK_NULL_HANDLE);
        m_imagesInFlight.assign(imageCount, VK_NULL_HANDLE);
        m_swapchainImageLayouts.assign(
            imageCount,
            VK_IMAGE_LAYOUT_UNDEFINED);

        VkSemaphoreCreateInfo semInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };

        for (VkSemaphore& semaphore : m_imageRenderFinished)
        {
            if (vkCreateSemaphore(
                m_device.device(),
                &semInfo,
                nullptr,
                &semaphore) != VK_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to create Vulkan render-finished semaphore.");
            }
        }
    }

    void VulkanBackend::destroyFrameSync()
    {
        VkDevice device = m_device.device();

        if (device == VK_NULL_HANDLE)
            return;

        for (FrameSync& fs : m_frameSync)
        {
            if (fs.imageAvailable != VK_NULL_HANDLE)
            {
                vkDestroySemaphore(
                    device,
                    fs.imageAvailable,
                    nullptr);

                fs.imageAvailable = VK_NULL_HANDLE;
            }

            if (fs.inFlightFence != VK_NULL_HANDLE)
            {
                vkDestroyFence(
                    device,
                    fs.inFlightFence,
                    nullptr);

                fs.inFlightFence = VK_NULL_HANDLE;
            }
        }

        m_frameSync.clear();
    }

    void VulkanBackend::destroySwapchainSync()
    {
        VkDevice device = m_device.device();

        if (device == VK_NULL_HANDLE)
            return;

        for (VkSemaphore semaphore : m_imageRenderFinished)
        {
            if (semaphore != VK_NULL_HANDLE)
            {
                vkDestroySemaphore(
                    device,
                    semaphore,
                    nullptr);
            }
        }

        m_imageRenderFinished.clear();
        m_imagesInFlight.clear();
        m_swapchainImageLayouts.clear();
    }

    void VulkanBackend::onSwapchainRecreated()
    {
        vkDeviceWaitIdle(m_device.device());
        destroySceneResources();
        m_pipelineManager.shutdown();

        m_swapchain.recreate();

        if (!m_swapchain.validForRendering())
            return;

        initSwapchainSync();
        m_pipelineManager.init(m_device.device());
        m_sceneFrameResources.resize(m_frameSync.size());
        m_imageStates.clear();
    }

    TextureFormat VulkanBackend::swapchainTextureFormat() const
    {
        switch (m_swapchain.format())
        {
        case VK_FORMAT_R8G8B8A8_UNORM:
            return TextureFormat::RGBA8_UNorm;
        case VK_FORMAT_R8G8B8A8_SRGB:
            return TextureFormat::RGBA8_SRGB;
        case VK_FORMAT_B8G8R8A8_UNORM:
            return TextureFormat::BGRA8_UNorm;
        case VK_FORMAT_B8G8R8A8_SRGB:
            return TextureFormat::BGRA8_SRGB;
        default:
            return TextureFormat::BGRA8_UNorm;
        }
    }

}

