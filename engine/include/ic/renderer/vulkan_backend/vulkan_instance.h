#pragma once
#include <vulkan/vulkan.h>
#include <span>

namespace ic
{
	class VulkanInstance
	{
	public:
		void init(bool enableValidation);
		void shutdown();

		VkInstance instance() const
		{
			return m_instance;
		}

        bool validationEnabled() const { return m_validationEnabled; }

	private:
		VkInstance m_instance = VK_NULL_HANDLE;
		VkDebugUtilsMessengerEXT m_debugMessenger =
			VK_NULL_HANDLE;
        bool m_validationEnabled = false;

		void createInstance();
		void setupValidationLayers();
		void setupDebugMessenger();
		
		std::vector<const char*> requiredExtensions() const;
		bool checkValidationLayerSupport() const;

		static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
			VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
			VkDebugUtilsMessageTypeFlagsEXT messageType,
			const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
			void* pUserData);

		static constexpr std::array<const char*, 1> m_validationLayers =
		{
			"VK_LAYER_KHRONOS_validation"
		};

		std::span<const char* const> validationLayers() const;
	};
}
