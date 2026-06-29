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
            m_computeQueue = VK_NULL_HANDLE;
            m_transferQueue = VK_NULL_HANDLE;
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


        if (queues.graphics == UINT32_MAX ||
            queues.compute == UINT32_MAX ||
            queues.transfer == UINT32_MAX ||
            queues.present == UINT32_MAX)
        {
            throw std::runtime_error("Invalid queue family indices in VulkanDeviceInfo.");
        }

        std::vector<uint32_t> uniqueFamilies;

        auto addUniqueFamily =
            [&uniqueFamilies](uint32_t family)
            {
                if (std::find(
                    uniqueFamilies.begin(),
                    uniqueFamilies.end(),
                    family) == uniqueFamilies.end())
                {
                    uniqueFamilies.push_back(family);
                }
            };

        addUniqueFamily(queues.graphics);
        addUniqueFamily(queues.compute);
        addUniqueFamily(queues.transfer);
        addUniqueFamily(queues.present);

        if (uniqueFamilies.empty())
        {
            throw std::runtime_error("No Vulkan queue families selected.");
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

        if (!m_info.supportedFeatures.dynamicRendering ||
            !m_info.supportedFeatures.synchronization2 ||
            !m_info.supportedFeatures.descriptorIndexing ||
            !m_info.supportedFeatures.bufferDeviceAddress)
        {
            throw std::runtime_error(
                "Selected Vulkan device does not support required renderer features.");
        }

        VkPhysicalDeviceFeatures deviceFeatures{};

        VkPhysicalDeviceDescriptorIndexingFeatures enabledDescriptorIndexing{};
        enabledDescriptorIndexing.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
        enabledDescriptorIndexing.shaderUniformBufferArrayNonUniformIndexing =
            m_info.descriptorIndexingFeatures.shaderUniformBufferArrayNonUniformIndexing;
        enabledDescriptorIndexing.shaderSampledImageArrayNonUniformIndexing =
            VK_TRUE;
        enabledDescriptorIndexing.shaderStorageBufferArrayNonUniformIndexing =
            VK_TRUE;
        enabledDescriptorIndexing.shaderStorageImageArrayNonUniformIndexing =
            VK_TRUE;
        enabledDescriptorIndexing.descriptorBindingSampledImageUpdateAfterBind =
            m_info.descriptorIndexingFeatures.descriptorBindingSampledImageUpdateAfterBind;
        enabledDescriptorIndexing.descriptorBindingStorageImageUpdateAfterBind =
            m_info.descriptorIndexingFeatures.descriptorBindingStorageImageUpdateAfterBind;
        enabledDescriptorIndexing.descriptorBindingStorageBufferUpdateAfterBind =
            m_info.descriptorIndexingFeatures.descriptorBindingStorageBufferUpdateAfterBind;
        enabledDescriptorIndexing.descriptorBindingUpdateUnusedWhilePending =
            VK_TRUE;
        enabledDescriptorIndexing.descriptorBindingPartiallyBound =
            VK_TRUE;
        enabledDescriptorIndexing.descriptorBindingVariableDescriptorCount =
            m_info.descriptorIndexingFeatures.descriptorBindingVariableDescriptorCount;
        enabledDescriptorIndexing.runtimeDescriptorArray =
            VK_TRUE;

        VkPhysicalDeviceBufferDeviceAddressFeatures enabledBda{};
        enabledBda.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
        enabledBda.bufferDeviceAddress =
            VK_TRUE;

        VkPhysicalDeviceTimelineSemaphoreFeatures enabledTimeline{};
        enabledTimeline.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
        enabledTimeline.timelineSemaphore =
            m_info.supportedFeatures.timelineSemaphore ? VK_TRUE : VK_FALSE;

        VkPhysicalDeviceVulkan13Features enabledVulkan13Features{};
        enabledVulkan13Features.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        enabledVulkan13Features.dynamicRendering =
            VK_TRUE;
        enabledVulkan13Features.synchronization2 =
            VK_TRUE;

#ifdef VK_EXT_descriptor_buffer
        VkPhysicalDeviceDescriptorBufferFeaturesEXT enabledDescriptorBuffer{};
        enabledDescriptorBuffer.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT;
        enabledDescriptorBuffer.descriptorBuffer =
            m_info.supportedFeatures.descriptorBuffer ? VK_TRUE : VK_FALSE;
#endif

        enabledDescriptorIndexing.pNext = &enabledBda;
        enabledBda.pNext = &enabledTimeline;
        enabledTimeline.pNext = &enabledVulkan13Features;

#ifdef VK_EXT_descriptor_buffer
        enabledVulkan13Features.pNext = &enabledDescriptorBuffer;
#endif

        std::vector<const char*> deviceExtensions =
        {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME
        };

#ifdef VK_EXT_descriptor_buffer
        if (m_info.supportedFeatures.descriptorBuffer)
        {
            deviceExtensions.push_back(VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME);
        }
#endif

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.pNext =
            &enabledDescriptorIndexing;

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
            m_info.queueFamilies.compute,
            0,
            &m_computeQueue);

        vkGetDeviceQueue(
            m_device,
            m_info.queueFamilies.transfer,
            0,
            &m_transferQueue);

        vkGetDeviceQueue(
            m_device,
            m_info.queueFamilies.present,
            0,
            &m_presentQueue);
    }
}
