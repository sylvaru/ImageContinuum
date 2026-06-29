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
        VkQueue presentQueue() const { return m_presentQueue; }

    private:
        void createLogicalDevice(
            const VulkanAdapter& adapter,
            VkSurfaceKHR surface);

        void getQueues(const VulkanAdapter& adapter);

    private:
        VkDevice m_device = VK_NULL_HANDLE;

        VulkanDeviceInfo m_info;
        VkQueue m_graphicsQueue;
        VkQueue m_presentQueue;
    };
}