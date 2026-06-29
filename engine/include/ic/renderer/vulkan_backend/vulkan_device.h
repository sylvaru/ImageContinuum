#pragma once
#include <vulkan/vulkan.h>
#include "vulkan_common.h"

namespace ic
{
    class VulkanAdapter;

    class VulkanDevice
    {
    public:

        void init(
            const VulkanAdapter& adapter,
            VkSurfaceKHR surface);

        void shutdown();

        VkDevice device() const { return m_device; }

        VkQueue graphicsQueue() const { return m_graphicsQueue; }
        VkQueue computeQueue() const { return m_computeQueue; }
        VkQueue transferQueue() const { return m_transferQueue; }
        VkQueue presentQueue() const { return m_presentQueue; }

        const VulkanDeviceInfo& info() const { return m_info; }

    private:
        void createLogicalDevice(
            const VulkanAdapter& adapter,
            VkSurfaceKHR surface);

        void getQueues(const VulkanAdapter& adapter);

    private:
        VkDevice m_device = VK_NULL_HANDLE;

        VulkanDeviceInfo m_info;
        VkQueue m_graphicsQueue = VK_NULL_HANDLE;
        VkQueue m_computeQueue = VK_NULL_HANDLE;
        VkQueue m_transferQueue = VK_NULL_HANDLE;
        VkQueue m_presentQueue = VK_NULL_HANDLE;
    };
}
