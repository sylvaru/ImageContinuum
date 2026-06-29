#pragma once
#include <vulkan/vulkan.h>
#include "vulkan_common.h"

namespace ic
{
	class VulkanAdapter
	{
	public:
		void init(
			VkInstance instance,
			VkSurfaceKHR surface);

		VkPhysicalDevice device() const
		{
			return m_device;
		}

		const VulkanDeviceInfo& info() const
		{
			return m_info;
		}


		SwapChainSupportDetails querySwapchainSupport(
			VkPhysicalDevice device) const;

	private:
		bool isSuitable(VkPhysicalDevice device);
		QueueFamilyIndices findQueueFamilies(
			VkPhysicalDevice device);


		void logDeviceInfo() const;

		VulkanDeviceInfo m_info;
		VkSurfaceKHR m_surface = VK_NULL_HANDLE;
		VkPhysicalDevice m_device = VK_NULL_HANDLE;
		std::vector<VkPhysicalDevice> m_candidates;
	};

}