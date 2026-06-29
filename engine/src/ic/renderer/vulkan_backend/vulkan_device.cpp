#include "ic/renderer/vulkan_backend/vulkan_device.h"
#include "ic/renderer/vulkan_backend/vulkan_adapter.h"

#include <spdlog/spdlog.h>

namespace ic
{
	void VulkanDevice::init(
		const VulkanAdapter& adapter, 
		VkSurfaceKHR surface)
	{
        m_info = adapter.info();
		createLogicalDevice(adapter, surface);
		getQueues(adapter);
              
	}
	void VulkanDevice::shutdown()
	{
        if (m_device != VK_NULL_HANDLE)
        {
            vkDestroyDevice(
                m_device,
                nullptr);

            m_device = VK_NULL_HANDLE;
            m_graphicsQueue = VK_NULL_HANDLE;
            m_presentQueue = VK_NULL_HANDLE;
        }
	}
    void VulkanDevice::createLogicalDevice(
        const VulkanAdapter& adapter,
        [[maybe_unused]] VkSurfaceKHR surface)
    {
        // Queue priorities
        float priority = 1.0f;

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;

        auto queues = m_info.queueFamilies;


        if (queues.graphics == UINT32_MAX || queues.present == UINT32_MAX)
        {
            throw std::runtime_error("Invalid queue family indices in VulkanDeviceInfo.");
        }

        std::vector<uint32_t> uniqueFamilies;

        uniqueFamilies.push_back(queues.graphics);

        if (queues.present != queues.graphics)
        {
            uniqueFamilies.push_back(queues.present);
        }

        for (uint32_t family : uniqueFamilies)
        {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = family;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &priority;

            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceVulkan13Features availableVulkan13Features{};
        availableVulkan13Features.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

        VkPhysicalDeviceFeatures2 availableFeatures{};
        availableFeatures.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        availableFeatures.pNext =
            &availableVulkan13Features;

        vkGetPhysicalDeviceFeatures2(
            adapter.device(),
            &availableFeatures);

        if (!availableVulkan13Features.dynamicRendering)
        {
            throw std::runtime_error(
                "Selected Vulkan device does not support dynamic rendering.");
        }

        VkPhysicalDeviceFeatures deviceFeatures{};

        VkPhysicalDeviceVulkan13Features enabledVulkan13Features{};
        enabledVulkan13Features.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        enabledVulkan13Features.dynamicRendering =
            VK_TRUE;

        std::vector<const char*> deviceExtensions =
        {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME
        };

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.pNext =
            &enabledVulkan13Features;

        createInfo.queueCreateInfoCount =
            static_cast<uint32_t>(queueCreateInfos.size());

        createInfo.pQueueCreateInfos =
            queueCreateInfos.data();

        createInfo.pEnabledFeatures =
            &deviceFeatures;

        createInfo.enabledExtensionCount =
            static_cast<uint32_t>(deviceExtensions.size());

        createInfo.ppEnabledExtensionNames =
            deviceExtensions.data();

        if (vkCreateDevice(
            adapter.device(),
            &createInfo,
            nullptr,
            &m_device) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Vulkan logical device.");
        }
    }
    void VulkanDevice::getQueues(
        [[maybe_unused]] const VulkanAdapter& adapter)
    {
        vkGetDeviceQueue(
            m_device,
            m_info.queueFamilies.graphics,
            0,
            &m_graphicsQueue);

        vkGetDeviceQueue(
            m_device,
            m_info.queueFamilies.present,
            0,
            &m_presentQueue);
    }
}
