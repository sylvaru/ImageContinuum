#include "ic/renderer/vulkan_backend/vulkan_upload_scheduler.h"

#include <algorithm>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace ic
{
    void VulkanUploadScheduler::init(
        const VulkanDevice& device,
        uint32_t framesInFlight)
    {
        m_device = &device;
        m_queue = device.transferQueue() != VK_NULL_HANDLE
            ? device.transferQueue()
            : device.graphicsQueue();
        const uint32_t uploadQueueFamily =
            m_queue == device.transferQueue()
                ? device.info().queueFamilies.transfer
                : device.info().queueFamilies.graphics;
        m_slots.resize(std::max(1u, framesInFlight));

        for (Slot& slot : m_slots)
        {
            VkCommandPoolCreateInfo poolInfo{
                VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
            poolInfo.queueFamilyIndex = uploadQueueFamily;
            poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            if (vkCreateCommandPool(
                    device.device(), &poolInfo, nullptr, &slot.pool) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create Vulkan upload command pool.");
            }
            VkCommandBufferAllocateInfo allocInfo{
                VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
            allocInfo.commandPool = slot.pool;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = 1;
            if (vkAllocateCommandBuffers(
                    device.device(), &allocInfo, &slot.commandBuffer) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to allocate Vulkan upload command buffer.");
            }
        }

        VkSemaphoreTypeCreateInfo type{
            VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo createInfo{
            VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        createInfo.pNext = &type;
        if (vkCreateSemaphore(
                device.device(), &createInfo, nullptr, &m_timeline) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Vulkan upload timeline.");
        }
        spdlog::info(
            "[VulkanUploadScheduler] Initialized (queueFamily={}, dedicatedTransfer={}, frames={})",
            uploadQueueFamily,
            uploadQueueFamily != device.info().queueFamilies.graphics,
            m_slots.size());
    }

    void VulkanUploadScheduler::shutdown()
    {
        if (!m_device)
        {
            return;
        }
        for (Slot& slot : m_slots)
        {
            if (slot.pool != VK_NULL_HANDLE)
            {
                vkDestroyCommandPool(m_device->device(), slot.pool, nullptr);
            }
        }
        if (m_timeline != VK_NULL_HANDLE)
        {
            vkDestroySemaphore(m_device->device(), m_timeline, nullptr);
        }
        m_slots.clear();
        m_timeline = VK_NULL_HANDLE;
        m_device = nullptr;
    }

    void VulkanUploadScheduler::beginFrame(uint32_t frameSlot)
    {
        std::scoped_lock lock(m_mutex);
        m_currentSlot = frameSlot;
        Slot& slot = m_slots[frameSlot];
        vkResetCommandPool(m_device->device(), slot.pool, 0);
        slot.recording = false;
        slot.hasCommands = false;
    }

    void VulkanUploadScheduler::record(
        const std::function<void(VkCommandBuffer)>& commands)
    {
        std::scoped_lock lock(m_mutex);
        Slot& slot = m_slots[m_currentSlot];
        if (!slot.recording)
        {
            VkCommandBufferBeginInfo beginInfo{
                VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (vkBeginCommandBuffer(slot.commandBuffer, &beginInfo) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to begin Vulkan upload batch.");
            }
            slot.recording = true;
        }
        commands(slot.commandBuffer);
        slot.hasCommands = true;
    }

    VulkanUploadDependency VulkanUploadScheduler::flush()
    {
        std::scoped_lock lock(m_mutex);
        Slot& slot = m_slots[m_currentSlot];
        if (!slot.hasCommands)
        {
            return {};
        }
        if (vkEndCommandBuffer(slot.commandBuffer) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to end Vulkan upload batch.");
        }
        slot.recording = false;

        const uint64_t value = m_nextValue++;
        VkTimelineSemaphoreSubmitInfo timeline{
            VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
        timeline.signalSemaphoreValueCount = 1;
        timeline.pSignalSemaphoreValues = &value;
        VkSubmitInfo submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submit.pNext = &timeline;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &slot.commandBuffer;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &m_timeline;
        if (vkQueueSubmit(m_queue, 1, &submit, VK_NULL_HANDLE) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to submit Vulkan upload batch.");
        }
        slot.hasCommands = false;
        return { m_timeline, value, m_queue };
    }
}
