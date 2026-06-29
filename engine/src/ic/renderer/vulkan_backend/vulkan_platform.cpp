#include "ic/common/ic_pch.h"
#include "ic/renderer/vulkan_backend/vulkan_platform.h"
#include "ic/interface/window.h"

#include <spdlog/spdlog.h>
#include <GLFW/glfw3.h>

namespace ic
{
	void VulkanPlatform::init(VkInstance instance, Window& window)
	{
		m_instance = instance;

		auto* glfwWindow = static_cast<GLFWwindow*>(window.getNativeHandle());

		VkResult result =
			glfwCreateWindowSurface(
				instance,
				glfwWindow,
				nullptr,
				&m_surface);

		if (result != VK_SUCCESS)
			throw std::runtime_error("Failed to create Vulkan surface");

		spdlog::info("[VulkanPlatform] Created presentation surface");
	}

	void VulkanPlatform::shutdown()
	{
		if (m_surface != VK_NULL_HANDLE)
		{
			vkDestroySurfaceKHR(
				m_instance,
				m_surface,
				nullptr);

			m_surface = VK_NULL_HANDLE;
		}

		spdlog::info(
			"[VulkanPlatform] Destroyed presentation surface.");
	}

}