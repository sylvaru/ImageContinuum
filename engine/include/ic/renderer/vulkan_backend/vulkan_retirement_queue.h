#pragma once

#include "ic/renderer/vulkan_backend/vulkan_resource_allocator.h"

#include <cstdint>
#include <mutex>
#include <vector>

namespace ic
{
    // Backend-local frame-slot retirement. beginFrame is called only after the
    // frame executor has observed completion for that slot, so recycling adds
    // no CPU or device-wide wait.
    class VulkanRetirementQueue final
    {
    public:
        void init(
            VkDevice device,
            VulkanResourceAllocator& allocator,
            uint32_t framesInFlight);
        void beginFrame(uint32_t frameSlot);
        void retire(VulkanBuffer&& buffer);
        void retire(VulkanTexture&& texture);
        void retireImageView(VkImageView view);
        void retireSampler(VkSampler sampler);
        void retireDescriptorPool(VkDescriptorPool pool);
        void retireAccelerationStructure(VkAccelerationStructureKHR as);
        void drain();

    private:
        struct Slot
        {
            std::vector<VulkanBuffer> buffers;
            std::vector<VulkanTexture> textures;
            std::vector<VkImageView> imageViews;
            std::vector<VkSampler> samplers;
            std::vector<VkDescriptorPool> descriptorPools;
            std::vector<VkAccelerationStructureKHR> accelerationStructures;
        };

        void recycle(Slot& slot);

        VkDevice m_device = VK_NULL_HANDLE;
        PFN_vkDestroyAccelerationStructureKHR m_destroyAccelerationStructure = nullptr;
        VulkanResourceAllocator* m_allocator = nullptr;
        std::vector<Slot> m_slots;
        uint32_t m_currentSlot = 0;
        std::mutex m_mutex;
    };
}
