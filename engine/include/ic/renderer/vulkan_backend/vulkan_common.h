#pragma once


namespace ic
{
    enum class DeviceCapability : uint32_t
    {
        DynamicRendering,
        Synchronization2,
        DescriptorIndexing,
        DescriptorBuffer,
        BufferDeviceAddress
    };

    struct VulkanFeatureSupport
    {
        bool dynamicRendering = false;
        bool synchronization2 = false;
        bool descriptorIndexing = false;
        bool descriptorBuffer = false;
        bool bufferDeviceAddress = false;
        bool timelineSemaphore = false;
    };

    struct QueueFamilyIndices
    {
        uint32_t graphics = UINT32_MAX;
        uint32_t compute = UINT32_MAX;
        uint32_t transfer = UINT32_MAX;
        uint32_t present = UINT32_MAX;

        bool isComplete() const
        {
            return graphics != UINT32_MAX &&
                present != UINT32_MAX &&
                compute != UINT32_MAX &&
                transfer != UINT32_MAX;
        }
    };

    struct SwapChainSupportDetails
    {
        VkSurfaceCapabilitiesKHR capabilities{};
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };

    struct VulkanDeviceInfo
    {

        VkPhysicalDeviceProperties properties{};

        VkPhysicalDeviceMemoryProperties memoryProperties{};

        VkPhysicalDeviceFeatures2 features{};

        QueueFamilyIndices queueFamilies;

        SwapChainSupportDetails swapchainSupport;

        VulkanFeatureSupport supportedFeatures;
    };


}