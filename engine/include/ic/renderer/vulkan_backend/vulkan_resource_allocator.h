#pragma once
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

namespace ic
{

	class VulkanResourceAllocator
	{
		void init(
			VkInstance,
			VkPhysicalDevice,
			VkDevice);

		void shutdown();
	public:
		VmaAllocator m_allocator;
	};
}