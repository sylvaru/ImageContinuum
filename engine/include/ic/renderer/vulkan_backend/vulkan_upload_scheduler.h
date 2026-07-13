#pragma once

#include "ic/renderer/vulkan_backend/vulkan_device.h"
#include "ic/renderer/vulkan_backend/vulkan_resource_allocator.h"

#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

namespace ic
{
    struct VulkanUploadDependency
    {
        VkSemaphore timeline = VK_NULL_HANDLE;
        uint64_t value = 0;
        VkQueue queue = VK_NULL_HANDLE;
    };

    // Batches runtime copies into one submission per frame. Resources used by
    // that submission are retired by frame slot after the frame fence proves
    // both the upload and all graph consumers have completed.
    class VulkanUploadScheduler final
    {
    public:
        void init(
            const VulkanDevice& device,
            uint32_t framesInFlight);
        void shutdown();
        void beginFrame(uint32_t frameSlot);
        void record(const std::function<void(VkCommandBuffer)>& commands);
        VulkanUploadDependency flush();

    private:
        struct Slot
        {
            VkCommandPool pool = VK_NULL_HANDLE;
            VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
            bool recording = false;
            bool hasCommands = false;
        };

        const VulkanDevice* m_device = nullptr;
        std::vector<Slot> m_slots;
        VkSemaphore m_timeline = VK_NULL_HANDLE;
        VkQueue m_queue = VK_NULL_HANDLE;
        uint64_t m_nextValue = 1;
        uint32_t m_currentSlot = 0;
        std::mutex m_mutex;
    };
}
