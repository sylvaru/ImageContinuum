#pragma once
#include <vulkan/vulkan.h>
#include <span>

namespace ic
{
    class VulkanAdapter;
    class VulkanDevice;
    class Window;

    enum class SwapchainState
    {
        Valid,
        OutOfDate,
        Minimized
    };


    class VulkanSwapchain
    {
    public:

        void init(
            const VulkanAdapter& adapter,
            VulkanDevice& device,
            VkSurfaceKHR surface,
            Window& window);

        void shutdown();

        void recreate();

        uint32_t acquireNextImage(
            VkSemaphore imageAvailable);

        bool present(
            VkQueue presentQueue,
            uint32_t imageIndex,
            VkSemaphore renderFinished);


        VkImage image(uint32_t index) const
        {
            return m_images[index];
        }

        const std::vector<VkImage>& images() const
        {
            return m_images;
        }

        VkImageView imageView(uint32_t index) const
        {
            return m_imageViews[index];
        }

        SwapchainState state() const
        {
            return m_state;
        }

        bool validForRendering() const
        {
            return m_state == SwapchainState::Valid;
        }

        bool vsyncEnabled() const
        {
            return m_vsync;
        }

        bool setVsyncEnabled(bool enabled)
        {
            if (m_vsync == enabled)
            {
                return false;
            }

            m_vsync = enabled;
            return true;
        }

        VkFormat format() const { return m_format; }
        VkExtent2D extent() const { return m_extent; }

    private:

        bool createSwapchain();
        void createImageViews();

        VkSurfaceFormatKHR chooseSurfaceFormat(
            std::span<const VkSurfaceFormatKHR> formats) const;

        VkPresentModeKHR choosePresentMode(
            std::span<const VkPresentModeKHR> modes) const;

        VkExtent2D chooseExtent(
            const VkSurfaceCapabilitiesKHR& capabilities,
            Window& window) const;

    private:

        const VulkanAdapter* m_adapter = nullptr;
        const VulkanDevice* m_device = nullptr;
        Window* m_window = nullptr;

        VkSurfaceKHR m_surface = VK_NULL_HANDLE;

        VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;

        std::vector<VkImage> m_images;
        std::vector<VkImageView> m_imageViews;

        VkFormat m_format{};
        VkExtent2D m_extent{};
        SwapchainState m_state = SwapchainState::Valid;
        bool m_vsync = true;

        uint32_t m_currentImage = 0;
    };

}
