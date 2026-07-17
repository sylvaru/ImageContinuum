#include "ic/common/ic_pch.h"
#include "ic/renderer/vulkan_backend/vulkan_instance.h"

#include <spdlog/spdlog.h>
#include <GLFW/glfw3.h>

namespace ic
{
	void VulkanInstance::init(bool enableValidation)
	{
		m_validationEnabled = enableValidation;
		spdlog::info("[VulkanInstance] Initializing...");

		createInstance();
		setupValidationLayers();
		setupDebugMessenger();

		spdlog::info("[VulkanInstance] Initialized.");
	}

    void VulkanInstance::shutdown()
    {
        if (m_debugMessenger != VK_NULL_HANDLE)
        {
            auto destroy =
                reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                    vkGetInstanceProcAddr(
                        m_instance,
                        "vkDestroyDebugUtilsMessengerEXT"));

            if (destroy)
            {
                destroy(
                    m_instance,
                    m_debugMessenger,
                    nullptr);
            }

            m_debugMessenger = VK_NULL_HANDLE;

            spdlog::info(
                "[VulkanInstance] Debug messenger destroyed.");
        }

        if (m_instance != VK_NULL_HANDLE)
        {
            vkDestroyInstance(
                m_instance,
                nullptr);

            m_instance = VK_NULL_HANDLE;

            spdlog::info(
                "[VulkanInstance] Instance destroyed.");
        }
    }

    void VulkanInstance::createInstance()
    {
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "ImageContinuum";
        appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);

        appInfo.pEngineName = "ImageContinuum";
        appInfo.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);

        appInfo.apiVersion = VK_API_VERSION_1_3;

        auto extensions = requiredExtensions();
        auto layers = validationLayers();

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        createInfo.enabledExtensionCount =
            static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames =
            extensions.data();

        createInfo.enabledLayerCount = m_validationEnabled
            ? static_cast<uint32_t>(layers.size()) : 0u;
        createInfo.ppEnabledLayerNames = m_validationEnabled
            ? layers.data() : nullptr;

        VkResult result =
            vkCreateInstance(
                &createInfo,
                nullptr,
                &m_instance);

        if (result != VK_SUCCESS)
        {
            throw std::runtime_error(
                "Failed to create Vulkan instance.");
        }

        spdlog::info(
            "[VulkanInstance] Vulkan {}.{}.{}",
            VK_API_VERSION_MAJOR(VK_API_VERSION_1_3),
            VK_API_VERSION_MINOR(VK_API_VERSION_1_3),
            VK_API_VERSION_PATCH(VK_API_VERSION_1_3));

        spdlog::info(
            "[VulkanInstance] Instance created.");
    }

    void VulkanInstance::setupValidationLayers()
    {
        if (!m_validationEnabled)
        {
            return;
        }

        if (!checkValidationLayerSupport())
        {
            throw std::runtime_error(
                "Requested Vulkan validation layers are unavailable.");
        }

        spdlog::info(
            "[VulkanInstance] Validation layers enabled.");

    }

    void VulkanInstance::setupDebugMessenger()
    {
        if (!m_validationEnabled)
        {
            return;
        }

        VkDebugUtilsMessengerCreateInfoEXT createInfo{};

        createInfo.sType =
            VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;

        createInfo.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;

        createInfo.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;

        createInfo.pfnUserCallback =
            debugCallback;

        auto func =
            reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(
                    m_instance,
                    "vkCreateDebugUtilsMessengerEXT"));

        if (!func)
        {
            throw std::runtime_error(
                "Failed to load vkCreateDebugUtilsMessengerEXT.");
        }

        if (func(
            m_instance,
            &createInfo,
            nullptr,
            &m_debugMessenger) != VK_SUCCESS)
        {
            throw std::runtime_error(
                "Failed to create Vulkan debug messenger.");
        }

        spdlog::info(
            "[VulkanInstance] Debug messenger created.");

    }

    std::vector<const char*> VulkanInstance::requiredExtensions() const
    {
        uint32_t count = 0;
        const char** glfwExtensions =
            glfwGetRequiredInstanceExtensions(&count);

        if (!glfwExtensions && count == 0)
        {
            throw std::runtime_error(
                "GLFW failed to provide Vulkan instance extensions.");
        }

        std::vector<const char*> extensions;
        extensions.reserve(count + 1);

        extensions.insert(
            extensions.end(),
            glfwExtensions,
            glfwExtensions + count
        );

        if (m_validationEnabled)
        {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    bool VulkanInstance::checkValidationLayerSupport() const
    {
        uint32_t layerCount = 0;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* required : m_validationLayers)
        {
            bool found = false;

            for (const auto& layer : availableLayers)
            {
                if (strcmp(required, layer.layerName) == 0)
                {
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                spdlog::error(
                    "[VulkanInstance] Missing validation layer: {}",
                    required);

                return false;
            }
        }

        return true;
    }

    VKAPI_ATTR VkBool32 VKAPI_CALL 
        VulkanInstance::debugCallback(
            VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
            [[maybe_unused]] VkDebugUtilsMessageTypeFlagsEXT messageType,
            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, 
            [[maybe_unused]] void* pUserData)
    {
        if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        {
            spdlog::error("[Vulkan Validation] {}", pCallbackData->pMessage);
        }
        else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        {
            spdlog::warn("[Vulkan Validation] {}", pCallbackData->pMessage);
        }
        else
        {
            spdlog::info("[Vulkan Validation] {}", pCallbackData->pMessage);
        }

        return VK_FALSE;
    }

    std::span<const char* const> VulkanInstance::validationLayers() const
    {
        return m_validationLayers;
    }


}
