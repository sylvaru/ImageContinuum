#include "ic/renderer/vulkan_backend/vulkan_swapchain.h"
#include "ic/renderer/vulkan_backend/vulkan_adapter.h"
#include "ic/renderer/vulkan_backend/vulkan_device.h"
#include "ic/interface/window.h"

#include <spdlog/spdlog.h>


namespace ic
{
	void VulkanSwapchain::init(
		const VulkanAdapter& adapter,
		VulkanDevice& device,
		VkSurfaceKHR surface,
		Window& window)
	{
		m_adapter = &adapter;
		m_device = &device;
		m_surface = surface;
		m_window = &window;

        if (!createSwapchain())
            return;

		createImageViews();

		spdlog::info(
			"[VulkanSwapchain] Created ({} images, {}x{})",
			m_images.size(),
			m_extent.width,
			m_extent.height);
	}

    bool VulkanSwapchain::present(
        VkQueue presentQueue,
        uint32_t imageIndex,
        VkSemaphore renderFinished)
    {
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType =
            VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount =
            1;

        presentInfo.pWaitSemaphores =
            &renderFinished;

        presentInfo.swapchainCount =
            1;

        presentInfo.pSwapchains =
            &m_swapchain;

        presentInfo.pImageIndices =
            &imageIndex;

        VkResult result =
            vkQueuePresentKHR(
                presentQueue,
                &presentInfo);

        switch (result)
        {
        case VK_SUCCESS:
            return true;

        case VK_SUBOPTIMAL_KHR:
        case VK_ERROR_OUT_OF_DATE_KHR:
            m_state = SwapchainState::OutOfDate;
            return false;

        default:
            throw std::runtime_error(
                "Failed to present swapchain image.");
        }
    }


    void VulkanSwapchain::recreate()
    {
        vkDeviceWaitIdle(m_device->device());

        shutdown();

        if (!createSwapchain())
            return;

        createImageViews();

        spdlog::info(
            "[VulkanSwapchain] Recreated ({}x{})",
            m_extent.width,
            m_extent.height);
    }


    uint32_t VulkanSwapchain::acquireNextImage(
        VkSemaphore imageAvailable)
    {
        if (m_state != SwapchainState::Valid)
            return UINT32_MAX;

        VkResult result =
            vkAcquireNextImageKHR(
                m_device->device(),
                m_swapchain,
                UINT64_MAX,
                imageAvailable,
                VK_NULL_HANDLE,
                &m_currentImage);

        switch (result)
        {
        case VK_SUCCESS:
        case VK_SUBOPTIMAL_KHR:
            return m_currentImage;

        case VK_ERROR_OUT_OF_DATE_KHR:
            m_state = SwapchainState::OutOfDate;
            return UINT32_MAX;

        default:
            throw std::runtime_error(
                "Failed to acquire swapchain image.");
        }
    }

    bool VulkanSwapchain::createSwapchain()
    {
        const auto& support =
            m_adapter->querySwapchainSupport(m_adapter->device());

        auto surfaceFormat =
            chooseSurfaceFormat(support.formats);

        auto presentMode =
            choosePresentMode(support.presentModes);

        auto extent =
            chooseExtent(
                support.capabilities,
                *m_window);

        if (extent.width == 0 || extent.height == 0)
        {
            m_state = SwapchainState::Minimized;
            m_extent = extent;
            return false;
        }

        uint32_t imageCount =
            support.capabilities.minImageCount + 1;

        if (support.capabilities.maxImageCount > 0 &&
            imageCount > support.capabilities.maxImageCount)
        {
            imageCount =
                support.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType =
            VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;

        createInfo.surface =
            m_surface;

        createInfo.minImageCount =
            imageCount;

        createInfo.imageFormat =
            surfaceFormat.format;

        createInfo.imageColorSpace =
            surfaceFormat.colorSpace;

        createInfo.imageExtent =
            extent;

        createInfo.imageArrayLayers =
            1;

        createInfo.imageUsage =
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT;

        const auto& queues =
            m_adapter->info().queueFamilies;

        uint32_t queueFamilyIndices[] =
        {
            queues.graphics,
            queues.present
        };

        if (queues.graphics != queues.present)
        {
            createInfo.imageSharingMode =
                VK_SHARING_MODE_CONCURRENT;

            createInfo.queueFamilyIndexCount =
                2;

            createInfo.pQueueFamilyIndices =
                queueFamilyIndices;
        }
        else
        {
            createInfo.imageSharingMode =
                VK_SHARING_MODE_EXCLUSIVE;
        }

        createInfo.preTransform =
            support.capabilities.currentTransform;

        createInfo.compositeAlpha =
            VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

        createInfo.presentMode =
            presentMode;

        createInfo.clipped =
            VK_TRUE;

        if (vkCreateSwapchainKHR(
            m_device->device(),
            &createInfo,
            nullptr,
            &m_swapchain) != VK_SUCCESS)
        {
            throw std::runtime_error(
                "Failed to create Vulkan swapchain.");
        }

        vkGetSwapchainImagesKHR(
            m_device->device(),
            m_swapchain,
            &imageCount,
            nullptr);

        m_images.resize(imageCount);

        vkGetSwapchainImagesKHR(
            m_device->device(),
            m_swapchain,
            &imageCount,
            m_images.data());

        m_format =
            surfaceFormat.format;

        m_extent = extent;

        m_state = SwapchainState::Valid;
        return true;
    }

    void VulkanSwapchain::createImageViews()
    {
        m_imageViews.resize(
            m_images.size());

        for (size_t i = 0; i < m_images.size(); ++i)
        {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType =
                VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;

            createInfo.image =
                m_images[i];

            createInfo.viewType =
                VK_IMAGE_VIEW_TYPE_2D;

            createInfo.format =
                m_format;

            createInfo.subresourceRange.aspectMask =
                VK_IMAGE_ASPECT_COLOR_BIT;

            createInfo.subresourceRange.baseMipLevel =
                0;

            createInfo.subresourceRange.levelCount =
                1;

            createInfo.subresourceRange.baseArrayLayer =
                0;

            createInfo.subresourceRange.layerCount =
                1;

            if (vkCreateImageView(
                m_device->device(),
                &createInfo,
                nullptr,
                &m_imageViews[i]) != VK_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to create swapchain image view.");
            }
        }
    }

    VkSurfaceFormatKHR VulkanSwapchain::chooseSurfaceFormat(
        std::span<const VkSurfaceFormatKHR> formats) const
    {
        for (const auto& format : formats)
        {
            if (format.format == VK_FORMAT_B8G8R8A8_UNORM &&
                format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            {
                return format;
            }
        }

        return formats.front();
    }

    VkPresentModeKHR VulkanSwapchain::choosePresentMode(
        [[maybe_unused]] std::span<const VkPresentModeKHR> modes) const
    {
        // FIFO is the only universally available Vulkan present mode that
        // guarantees no tearing.
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D VulkanSwapchain::chooseExtent(
        const VkSurfaceCapabilitiesKHR& capabilities,
        [[maybe_unused]] Window& window) const
    {
        if (capabilities.currentExtent.width !=
            std::numeric_limits<uint32_t>::max())
        {
            return capabilities.currentExtent;
        }

        VkExtent2D extent
        {
            static_cast<uint32_t>(m_window->getWidth()),
            static_cast<uint32_t>(m_window->getHeight())
        };

        extent.width =
            std::clamp(
                extent.width,
                capabilities.minImageExtent.width,
                capabilities.maxImageExtent.width);

        extent.height =
            std::clamp(
                extent.height,
                capabilities.minImageExtent.height,
                capabilities.maxImageExtent.height);

        return extent;
    }

    void VulkanSwapchain::shutdown()
    {
        VkDevice device = m_device->device();

        for (VkImageView view : m_imageViews)
        {
            vkDestroyImageView(
                device,
                view,
                nullptr);
        }

        m_imageViews.clear();

        if (m_swapchain != VK_NULL_HANDLE)
        {
            vkDestroySwapchainKHR(
                device,
                m_swapchain,
                nullptr);

            m_swapchain = VK_NULL_HANDLE;
        }
    }
}
