#include "ic/common/ic_pch.h"
#include "ic/renderer/vulkan_backend/vulkan_adapter.h"
#include <spdlog/spdlog.h>

namespace ic
{
	void VulkanAdapter::init(VkInstance instance, VkSurfaceKHR surface)
	{
		m_surface = surface;

		uint32_t count = 0;
		vkEnumeratePhysicalDevices(instance, &count, nullptr);

		if (count == 0)
			throw std::runtime_error("No Vulkan physical device found");

		std::vector<VkPhysicalDevice> candidates(count);

		vkEnumeratePhysicalDevices(
			instance,
			&count,
			candidates.data());

		spdlog::info(
			"[Vulkan] Found {} physical devices", count);

		// Find best device
		for (const auto& device : candidates)
		{
			if (isSuitable(device))
			{
				m_device = device;
				break;
			}
		}

		if (m_device == VK_NULL_HANDLE)
			throw std::runtime_error("No suitable Vulkan GPU found");

		// Cache info
		vkGetPhysicalDeviceProperties(
			m_device,
			&m_info.properties);

		vkGetPhysicalDeviceMemoryProperties(
			m_device,
			&m_info.memoryProperties);

		m_info.features.sType =
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

		vkGetPhysicalDeviceFeatures2(
			m_device,
			&m_info.features);

		m_info.queueFamilies =
			findQueueFamilies(m_device);

		logDeviceInfo();
	}

	bool VulkanAdapter::isSuitable(VkPhysicalDevice device)
	{
		auto queues =
			findQueueFamilies(device);

		if (!queues.isComplete())
			return false;

		auto swapchain =
			querySwapchainSupport(device);

		if (swapchain.formats.empty())
			return false;

		if (swapchain.presentModes.empty())
			return false;

		return true;
	}

	QueueFamilyIndices VulkanAdapter::findQueueFamilies(VkPhysicalDevice device) 
	{
		QueueFamilyIndices indices;

		uint32_t count = 0;

		vkGetPhysicalDeviceQueueFamilyProperties(
			device,
			&count,
			nullptr);

		std::vector<VkQueueFamilyProperties> queues(count);

		vkGetPhysicalDeviceQueueFamilyProperties(
			device,
			&count,
			queues.data());

		for (uint32_t i = 0; i < count; ++i)
		{
			if (queues[i].queueFlags &
				VK_QUEUE_GRAPHICS_BIT)
			{
				indices.graphics = i;
			}

			if (queues[i].queueFlags &
				VK_QUEUE_COMPUTE_BIT)
			{
				indices.compute = i;
			}

			if (queues[i].queueFlags &
				VK_QUEUE_TRANSFER_BIT)
			{
				indices.transfer = i;
			}

			VkBool32 presentSupport = false;

			vkGetPhysicalDeviceSurfaceSupportKHR(
				device,
				i,
				m_surface,
				&presentSupport);

			if (presentSupport)
			{
				indices.present = i;
			}
		}

		return indices;
	}

	SwapChainSupportDetails 
		VulkanAdapter::querySwapchainSupport(VkPhysicalDevice device) const
	{
		SwapChainSupportDetails details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
			device,
			m_surface,
			&details.capabilities);

		uint32_t count = 0;

		vkGetPhysicalDeviceSurfaceFormatsKHR(
			device,
			m_surface,
			&count,
			nullptr);

		details.formats.resize(count);

		vkGetPhysicalDeviceSurfaceFormatsKHR(
			device,
			m_surface,
			&count,
			details.formats.data());

		count = 0;

		vkGetPhysicalDeviceSurfacePresentModesKHR(
			device,
			m_surface,
			&count,
			nullptr);

		details.presentModes.resize(count);

		vkGetPhysicalDeviceSurfacePresentModesKHR(
			device,
			m_surface,
			&count,
			details.presentModes.data());

		return details;
	}

	void VulkanAdapter::logDeviceInfo() const
	{
		const auto& props = m_info.properties;

		spdlog::info("----------- Vulkan Adapter -----------");

		spdlog::info(
			"Name: {}",
			props.deviceName);

		spdlog::info(
			"Vendor ID: {}",
			props.vendorID);

		spdlog::info(
			"Device ID: {}",
			props.deviceID);

		spdlog::info(
			"API Version: {}.{}.{}",
			VK_API_VERSION_MAJOR(props.apiVersion),
			VK_API_VERSION_MINOR(props.apiVersion),
			VK_API_VERSION_PATCH(props.apiVersion));

		spdlog::info(
			"Driver Version: {}",
			props.driverVersion);

		spdlog::info(
			"Graphics Queue: {}",
			m_info.queueFamilies.graphics);

		spdlog::info(
			"Compute Queue: {}",
			m_info.queueFamilies.compute);

		spdlog::info(
			"Transfer Queue: {}",
			m_info.queueFamilies.transfer);

		spdlog::info(
			"Present Queue: {}",
			m_info.queueFamilies.present);

		auto memory =
			static_cast<double>(
				m_info.memoryProperties.memoryHeaps[0].size)
			/ (1024.0 * 1024.0 * 1024.0);

		spdlog::info(
			"VRAM: {:.2f} GB",
			memory);

		spdlog::info("--------------------------------------");
	}
}
