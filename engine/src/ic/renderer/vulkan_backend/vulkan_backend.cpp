#include "ic/renderer/vulkan_backend/vulkan_backend.h"
#include "ic/renderer/renderer_specification.h"
#include "ic/interface/window.h"
#include "ic/core/frame_context.h"

#include <spdlog/spdlog.h>

namespace ic
{
	void VulkanBackend::initialize(
		const RendererSpecification& spec,
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

		m_swapchain.init(
			m_adapter,
			m_device,
			m_platform.surface(),
			window);

        const uint32_t framesInFlight =
            spec.framesInFlight == 0 ? 1 : spec.framesInFlight;

        const uint32_t workerSlots =
            workerCount == 0 ? 1 : workerCount;

        initFrameSync(spec);

		m_commandSystem.init(
			m_device.device(),
			m_adapter.info().queueFamilies.graphics,
			framesInFlight,
			workerSlots);

	}

	void VulkanBackend::shutdown()
	{
        if (m_device.device() != VK_NULL_HANDLE)
        {
            vkDeviceWaitIdle(m_device.device());
        }

        destroySwapchainSync();
        destroyFrameSync();

        m_commandSystem.shutdown();
        m_swapchain.shutdown();
        m_device.shutdown();
        m_platform.shutdown();
        m_instance.shutdown();

		spdlog::info("[VulkanBackend] Shutdown");
	}

    void VulkanBackend::execute(
        const CompiledGraphPlan& plan,
        const FrameContext& ctx)
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

        // Begin frame command recording
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        beginInfo.pInheritanceInfo = nullptr;

        VkCommandBuffer cmd =
            m_commandSystem.beginFrameCommandBuffer(
                frameSlot, 0);


        vkBeginCommandBuffer(cmd, &beginInfo);

        // Execute compiled graph plan
        executeGraph(plan, ctx, cmd, swapchainImage);

        vkEndCommandBuffer(cmd);

        VkSemaphore renderFinished =
            m_imageRenderFinished[imageIndex];

        // Submit
        submitFrame(
            cmd,
            frameSync,
            renderFinished);

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

    void VulkanBackend::submitFrame(
        VkCommandBuffer cmd,
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

        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;

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
        VkCommandBuffer cmd,
        VkImage swapchainImage)
    {
        // executionOrder is already topologically sorted
        for (GraphNodeId nodeId : plan.executionOrder)
        {
            const ExecutionNode& node =
                plan.nodes[nodeId];

            // Apply barriers for this node
            applyBarriers(
                cmd,
                plan.barriers,
                plan.resources,
                node,
                swapchainImage);

            // Dispatch node work
            dispatchNode(
                node,
                ctx,
                cmd,
                swapchainImage);
        }
    }

    void VulkanBackend::applyBarriers(
        VkCommandBuffer cmd,
        std::span<const ResourceBarrier> barriers,
        std::span<const GraphResource> resources,
        const ExecutionNode& node,
        VkImage swapchainImage)
    {
        for (const ResourceBarrier& barrier : barriers)
        {
            if (barrier.toNode != node.nodeId)
                continue;


            recordBarrier(
                cmd,
                barrier,
                resources,
                swapchainImage);
        }
    }

    void VulkanBackend::recordBarrier(
        VkCommandBuffer cmd,
        const ResourceBarrier& barrier,
        std::span<const GraphResource> resources,
        VkImage swapchainImage)
    {

        VkImageMemoryBarrier vkBarrier{};
        vkBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;

        const GraphResource& resource =
            resources[barrier.resource];

        VkImage image = VK_NULL_HANDLE;

        if (resource.ownership == ResourceOwnership::Imported)
        {
            switch (resource.imported)
            {
            case ImportedResource::Swapchain:
                image = swapchainImage;
                break;
            }
        }
        else
        {
            // TODO
            return;
        }
        vkBarrier.image = image;

        const VkImageLayout trackedLayout =
            getOrInitImageLayout(image);

        vkBarrier.oldLayout = trackedLayout;
        vkBarrier.newLayout = usageToLayout(barrier.newUsage);

        vkBarrier.srcAccessMask =
            trackedLayout == VK_IMAGE_LAYOUT_UNDEFINED
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

        if (trackedLayout == VK_IMAGE_LAYOUT_UNDEFINED)
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

        m_imageStates[image].layout = vkBarrier.newLayout;
        m_imageStates[image].access = barrier.toAccess;
    }

    void VulkanBackend::dispatchNode(
        const ExecutionNode& node,
        const FrameContext& ctx,
        VkCommandBuffer cmd,
        [[maybe_unused]] VkImage swapchainImage)
    {
        switch (node.type)
        {
        case GraphNodeType::Graphics:
            executeGraphicsNode(node, ctx, cmd, swapchainImage);
            break;

        case GraphNodeType::Compute:
            //executeComputeNode(node, ctx, cmd);
            break;

        case GraphNodeType::Transfer:
            //executeTransferNode(node, ctx, cmd);
            break;

        case GraphNodeType::Present:
            break;
        }
    }

    void VulkanBackend::executeGraphicsNode(
        [[maybe_unused]] const ExecutionNode& node,
        [[maybe_unused]] const FrameContext& ctx,
        [[maybe_unused]] VkCommandBuffer cmd,
        [[maybe_unused]] VkImage swapchainImage)
    {

        VkClearValue clear{};
        clear.color = { { 1.0f, 0.0f, 1.0f, 1.0f } };

        VkRenderingAttachmentInfo colorAttachment{};
        colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttachment.imageView = m_swapchain.imageView(m_currentSwapchainImage);
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.clearValue = clear;

        VkRenderingInfo renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderingInfo.renderArea = { {0, 0}, m_swapchain.extent() };
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;

        vkCmdBeginRendering(cmd, &renderingInfo);


        vkCmdEndRendering(cmd);

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
    }

    void VulkanBackend::onSwapchainRecreated()
    {
        initSwapchainSync();
        m_imageStates.clear();
    }

}

