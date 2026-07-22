#include "ic/renderer/vulkan_backend/vulkan_retirement_queue.h"

#include <algorithm>
#include <stdexcept>

namespace ic
{
    void VulkanRetirementQueue::init(
        VkDevice device,
        VulkanResourceAllocator& allocator,
        uint32_t framesInFlight)
    {
        m_device = device;
        m_destroyAccelerationStructure = reinterpret_cast<
            PFN_vkDestroyAccelerationStructureKHR>(
                vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR"));
        m_allocator = &allocator;
        m_slots.resize(std::max(1u, framesInFlight));
    }

    void VulkanRetirementQueue::beginFrame(uint32_t frameSlot)
    {
        std::scoped_lock lock(m_mutex);
        if (frameSlot >= m_slots.size())
        {
            throw std::out_of_range("Vulkan retirement frame slot is out of range.");
        }
        m_currentSlot = frameSlot;
        recycle(m_slots[frameSlot]);
    }

    void VulkanRetirementQueue::retire(VulkanBuffer&& buffer)
    {
        std::scoped_lock lock(m_mutex);
        if (buffer) m_slots[m_currentSlot].buffers.emplace_back(std::move(buffer));
    }

    void VulkanRetirementQueue::retire(VulkanTexture&& texture)
    {
        std::scoped_lock lock(m_mutex);
        if (texture) m_slots[m_currentSlot].textures.emplace_back(std::move(texture));
    }

    void VulkanRetirementQueue::retireImageView(VkImageView view)
    {
        std::scoped_lock lock(m_mutex);
        if (view != VK_NULL_HANDLE) m_slots[m_currentSlot].imageViews.push_back(view);
    }

    void VulkanRetirementQueue::retireSampler(VkSampler sampler)
    {
        std::scoped_lock lock(m_mutex);
        if (sampler != VK_NULL_HANDLE) m_slots[m_currentSlot].samplers.push_back(sampler);
    }

    void VulkanRetirementQueue::retireDescriptorPool(VkDescriptorPool pool)
    {
        std::scoped_lock lock(m_mutex);
        if (pool != VK_NULL_HANDLE) m_slots[m_currentSlot].descriptorPools.push_back(pool);
    }

    void VulkanRetirementQueue::retireAccelerationStructure(
        VkAccelerationStructureKHR accelerationStructure)
    {
        if (accelerationStructure == VK_NULL_HANDLE)
            return;
        std::scoped_lock lock(m_mutex);
        m_slots[m_currentSlot].accelerationStructures.push_back(
            accelerationStructure);
    }

    void VulkanRetirementQueue::drain()
    {
        std::scoped_lock lock(m_mutex);
        for (Slot& slot : m_slots) recycle(slot);
        m_slots.clear();
        m_allocator = nullptr;
        m_device = VK_NULL_HANDLE;
    }

    void VulkanRetirementQueue::recycle(Slot& slot)
    {
        for (VkAccelerationStructureKHR as : slot.accelerationStructures)
            if (m_destroyAccelerationStructure)
                m_destroyAccelerationStructure(m_device, as, nullptr);
        slot.accelerationStructures.clear();
        // Descriptor objects and views must die before their backing images.
        for (VkDescriptorPool pool : slot.descriptorPools)
            vkDestroyDescriptorPool(m_device, pool, nullptr);
        for (VkImageView view : slot.imageViews)
            vkDestroyImageView(m_device, view, nullptr);
        for (VkSampler sampler : slot.samplers)
            vkDestroySampler(m_device, sampler, nullptr);
        for (VulkanTexture& texture : slot.textures)
            m_allocator->destroyTexture(texture);
        for (VulkanBuffer& buffer : slot.buffers)
            m_allocator->destroyBuffer(buffer);
        slot.descriptorPools.clear();
        slot.imageViews.clear();
        slot.samplers.clear();
        slot.textures.clear();
        slot.buffers.clear();
    }
}
