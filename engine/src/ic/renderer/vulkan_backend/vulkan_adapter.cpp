#include "ic/common/ic_pch.h"
#include "ic/renderer/vulkan_backend/vulkan_adapter.h"
#include <spdlog/spdlog.h>

#include <unordered_set>

namespace
{
	bool hasExtension(
		const std::unordered_set<std::string>& extensions,
		const char* name)
	{
		return extensions.contains(name);
	}

	std::unordered_set<std::string> enumerateDeviceExtensions(
		VkPhysicalDevice device)
	{
		uint32_t count = 0;
		vkEnumerateDeviceExtensionProperties(
			device,
			nullptr,
			&count,
			nullptr);

		std::vector<VkExtensionProperties> properties(count);
		vkEnumerateDeviceExtensionProperties(
			device,
			nullptr,
			&count,
			properties.data());

		std::unordered_set<std::string> extensions;
		extensions.reserve(properties.size());

		for (const VkExtensionProperties& property : properties)
		{
			extensions.emplace(property.extensionName);
		}

		return extensions;
	}
}

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

		m_info.descriptorIndexingFeatures.sType =
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;

		m_info.bufferDeviceAddressFeatures.sType =
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;

		m_info.vulkan13Features.sType =
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

		m_info.timelineSemaphoreFeatures.sType =
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;

#ifdef VK_EXT_descriptor_buffer
		m_info.descriptorBufferFeatures.sType =
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT;
#endif

		m_info.features.pNext = &m_info.descriptorIndexingFeatures;
		m_info.descriptorIndexingFeatures.pNext = &m_info.bufferDeviceAddressFeatures;
		m_info.bufferDeviceAddressFeatures.pNext = &m_info.vulkan13Features;
		m_info.vulkan13Features.pNext = &m_info.timelineSemaphoreFeatures;

#ifdef VK_EXT_descriptor_buffer
		m_info.timelineSemaphoreFeatures.pNext = &m_info.descriptorBufferFeatures;
#endif

		vkGetPhysicalDeviceFeatures2(
			m_device,
			&m_info.features);

		const auto extensions =
			enumerateDeviceExtensions(m_device);

		m_info.supportedFeatures.dynamicRendering =
			m_info.vulkan13Features.dynamicRendering == VK_TRUE;

		m_info.supportedFeatures.synchronization2 =
			m_info.vulkan13Features.synchronization2 == VK_TRUE;

		m_info.supportedFeatures.timelineSemaphore =
			m_info.timelineSemaphoreFeatures.timelineSemaphore == VK_TRUE;

		m_info.supportedFeatures.bufferDeviceAddress =
			m_info.bufferDeviceAddressFeatures.bufferDeviceAddress == VK_TRUE;

		m_info.supportedFeatures.descriptorIndexing =
			m_info.descriptorIndexingFeatures.runtimeDescriptorArray == VK_TRUE &&
			m_info.descriptorIndexingFeatures.descriptorBindingPartiallyBound == VK_TRUE &&
			m_info.descriptorIndexingFeatures.descriptorBindingUpdateUnusedWhilePending == VK_TRUE &&
			m_info.descriptorIndexingFeatures.shaderSampledImageArrayNonUniformIndexing == VK_TRUE &&
			m_info.descriptorIndexingFeatures.shaderStorageBufferArrayNonUniformIndexing == VK_TRUE &&
			m_info.descriptorIndexingFeatures.shaderStorageImageArrayNonUniformIndexing == VK_TRUE;

#ifdef VK_EXT_descriptor_buffer
		m_info.supportedFeatures.descriptorBuffer =
			hasExtension(extensions, VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME) &&
			m_info.descriptorBufferFeatures.descriptorBuffer == VK_TRUE;
#endif

#ifdef VK_KHR_maintenance5
		m_info.supportedFeatures.maintenance5 =
			hasExtension(extensions, VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
#endif

#ifdef VK_KHR_extended_flags
		m_info.supportedFeatures.extendedFlags =
			hasExtension(extensions, VK_KHR_EXTENDED_FLAGS_EXTENSION_NAME);
#endif

#ifdef VK_EXT_descriptor_heap
		const bool descriptorHeapDependencySatisfied =
			VK_API_VERSION_MAJOR(m_info.properties.apiVersion) > 1 ||
			(VK_API_VERSION_MAJOR(m_info.properties.apiVersion) == 1 &&
			 VK_API_VERSION_MINOR(m_info.properties.apiVersion) >= 4) ||
			m_info.supportedFeatures.maintenance5 ||
			m_info.supportedFeatures.extendedFlags;

		m_info.supportedFeatures.descriptorHeapAvailable =
			hasExtension(extensions, VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME) &&
			descriptorHeapDependencySatisfied;

		if (m_info.supportedFeatures.descriptorHeapAvailable)
		{
			spdlog::warn(
				"[VulkanAdapter] VK_EXT_descriptor_heap is advertised, but it is disabled because the current validation/runtime stack rejects its feature sType. Falling back to descriptor buffer/indexing.");
		}

		m_info.supportedFeatures.descriptorHeap = false;
#endif

#ifdef VK_EXT_descriptor_buffer
		m_info.descriptorBufferProperties.sType =
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT;
#endif

#ifdef VK_EXT_descriptor_buffer
		VkPhysicalDeviceProperties2 properties2{};
		properties2.sType =
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;

		properties2.pNext = &m_info.descriptorBufferProperties;

		vkGetPhysicalDeviceProperties2(
			m_device,
			&properties2);
#endif

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

		VkPhysicalDeviceVulkan13Features vulkan13{};
		vulkan13.sType =
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

		VkPhysicalDeviceDescriptorIndexingFeatures indexing{};
		indexing.sType =
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;

		VkPhysicalDeviceBufferDeviceAddressFeatures bda{};
		bda.sType =
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;

		VkPhysicalDeviceFeatures2 features{};
		features.sType =
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
		features.pNext = &vulkan13;
		vulkan13.pNext = &indexing;
		indexing.pNext = &bda;

		vkGetPhysicalDeviceFeatures2(
			device,
			&features);

		if (!vulkan13.dynamicRendering ||
			!vulkan13.synchronization2 ||
			!indexing.runtimeDescriptorArray ||
			!indexing.descriptorBindingPartiallyBound ||
			!bda.bufferDeviceAddress)
		{
			return false;
		}

		return true;
	}

	QueueFamilyIndices VulkanAdapter::findQueueFamilies(VkPhysicalDevice device) const
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

		spdlog::info(
			"Features | dynamicRendering={} sync2={} timeline={} descriptorIndexing={} descriptorHeap={} descriptorHeapAvailable={} descriptorBuffer={} bufferDeviceAddress={} maintenance5={} extendedFlags={}",
			m_info.supportedFeatures.dynamicRendering,
			m_info.supportedFeatures.synchronization2,
			m_info.supportedFeatures.timelineSemaphore,
			m_info.supportedFeatures.descriptorIndexing,
			m_info.supportedFeatures.descriptorHeap,
			m_info.supportedFeatures.descriptorHeapAvailable,
			m_info.supportedFeatures.descriptorBuffer,
			m_info.supportedFeatures.bufferDeviceAddress,
			m_info.supportedFeatures.maintenance5,
			m_info.supportedFeatures.extendedFlags);

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
