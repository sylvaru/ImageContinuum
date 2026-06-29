#include "ic/renderer/vulkan_backend/vulkan_command_system.h"


#include <spdlog/spdlog.h>


namespace ic
{
    void VulkanCommandSystem::init(
        VkDevice device,
        uint32_t graphicsQueueFamily,
        uint32_t maxFramesInFlight,
        uint32_t maxWorkers)
    {
        m_device = device;
        m_graphicsQueueFamily = graphicsQueueFamily;

        m_frames.resize(maxFramesInFlight);

        for (auto& frame : m_frames)
        {
            frame.workerPools.resize(maxWorkers);

            for (auto& worker : frame.workerPools)
            {
                VkCommandPoolCreateInfo poolInfo{};
                poolInfo.sType =
                    VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;

                poolInfo.queueFamilyIndex =
                    graphicsQueueFamily;

                poolInfo.flags =
                    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

                vkCreateCommandPool(
                    m_device,
                    &poolInfo,
                    nullptr,
                    &worker.pool);
            }
        }

        // fence for immediateSubmit
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType =
            VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

        fenceInfo.flags =
            VK_FENCE_CREATE_SIGNALED_BIT;

        vkCreateFence(
            m_device,
            &fenceInfo,
            nullptr,
            &m_immediateFence);

        spdlog::info("[VulkanCommandSystem] Initialized (frames={}, workers={})",
            maxFramesInFlight,
            maxWorkers);
    }

    void VulkanCommandSystem::shutdown()
    {
        vkDeviceWaitIdle(m_device);

        for (auto& frame : m_frames)
        {
            for (auto& worker : frame.workerPools)
            {
                if (worker.pool != VK_NULL_HANDLE)
                {
                    vkDestroyCommandPool(
                        m_device,
                        worker.pool,
                        nullptr);
                }
            }
        }

        if (m_immediateFence)
        {
            vkDestroyFence(
                m_device,
                m_immediateFence,
                nullptr);
        }

        m_frames.clear();
    }

    void VulkanCommandSystem::beginFrame(uint32_t frameIndex)
    {
        for (auto& worker : m_frames[frameIndex].workerPools)
        {
            vkResetCommandPool(
                m_device,
                worker.pool,
                0);
        }
    }

    VkCommandBuffer VulkanCommandSystem::beginFrameCommandBuffer(
        uint32_t frameIndex,
        uint32_t workerIndex)
    {
        FrameData& frame =
            m_frames[frameIndex % m_frames.size()];

        WorkerPool& pool =
            frame.workerPools[workerIndex];

        if (pool.primary == VK_NULL_HANDLE)
        {
            VkCommandBufferAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.commandPool = pool.pool;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = 1;

            vkAllocateCommandBuffers(
                m_device,
                &allocInfo,
                &pool.primary);
        }

        return pool.primary;
    }

    VkCommandBuffer VulkanCommandSystem::allocateCommandBuffer(
        uint32_t frameIndex,
        uint32_t workerIndex)
    {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType =
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;

        allocInfo.commandPool =
            m_frames[frameIndex]
            .workerPools[workerIndex]
            .pool;

        allocInfo.level =
            VK_COMMAND_BUFFER_LEVEL_PRIMARY;

        allocInfo.commandBufferCount =
            1;

        VkCommandBuffer cmd;
        vkAllocateCommandBuffers(
            m_device,
            &allocInfo,
            &cmd);

        return cmd;
    }

    void VulkanCommandSystem::immediateSubmit(
        VkQueue queue,
        std::function<void(VkCommandBuffer cmd)>&& function)
    {
        vkWaitForFences(
            m_device,
            1,
            &m_immediateFence,
            VK_TRUE,
            UINT64_MAX);

        vkResetFences(
            m_device,
            1,
            &m_immediateFence);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType =
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;

        poolInfo.queueFamilyIndex =
            m_graphicsQueueFamily;

        poolInfo.flags =
            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        VkCommandPool pool;
        vkCreateCommandPool(
            m_device,
            &poolInfo,
            nullptr,
            &pool);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType =
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level =
            VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool =
            pool;
        allocInfo.commandBufferCount =
            1;

        VkCommandBuffer cmd;
        vkAllocateCommandBuffers(
            m_device,
            &allocInfo,
            &cmd);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType =
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags =
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(cmd, &beginInfo);

        function(cmd);

        vkEndCommandBuffer(cmd);

        VkSubmitInfo submitInfo{};
        submitInfo.sType =
            VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount =
            1;
        submitInfo.pCommandBuffers =
            &cmd;

        vkQueueSubmit(
            queue,
            1,
            &submitInfo,
            m_immediateFence);

        vkWaitForFences(
            m_device,
            1,
            &m_immediateFence,
            VK_TRUE,
            UINT64_MAX);

        vkFreeCommandBuffers(
            m_device,
            pool,
            1,
            &cmd);

        vkDestroyCommandPool(
            m_device,
            pool,
            nullptr);
    }
}
