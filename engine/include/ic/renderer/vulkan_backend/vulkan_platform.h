#pragma once
#include <vulkan/vulkan.h>

namespace ic
{
	class Window;


	class VulkanPlatform
	{
	public:
		void init(VkInstance instance, Window& window);
		void shutdown();
		VkSurfaceKHR surface() const
		{
			return m_surface;
		}

	private:
		VkSurfaceKHR m_surface = VK_NULL_HANDLE;
		VkInstance m_instance = VK_NULL_HANDLE;
	};
}