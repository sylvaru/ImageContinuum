#include "ic/renderer/vulkan_backend/vulkan_command_system.h"


#include <spdlog/spdlog.h>
#include <stdexcept>


namespace ic
{
    namespace
    {
        uint32_t queueIndex(QueueType queue)
        {
            return static_cast<uint32_t>(queue);
        }

        uint32_t queueFamily(
            const QueueFamilyIndices& families,
            QueueType queue)
        {
            switch (queue)
            {
            case QueueType::Compute: return families.compute;
            case QueueType::Transfer: return families.transfer;
            case QueueType::Graphics: return families.graphics;
            }
            return families.graphics;
        }
    }

    void VulkanCommandSystem::init(
        VkDevice device,
        const QueueFamilyIndices& queueFamilies,
        uint32_t maxFramesInFlight,
        uint32_t maxWorkers)
    {
        m_device = device;
        m_queueFamilies = queueFamilies;

        m_frames.resize(maxFramesInFlight);

        for (auto& frame : m_frames)
        {
            for (uint32_t queue = 0; queue < frame.queuePools.size(); ++queue)
            {
                auto& workers = frame.queuePools[queue];
                workers.resize(maxWorkers);
                for (std::unique_ptr<WorkerPool>& workerPtr : workers)
                {
                    workerPtr = std::make_unique<WorkerPool>();
                    WorkerPool& worker = *workerPtr;

                VkCommandPoolCreateInfo poolInfo{};
                poolInfo.sType =
                    VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;

                    poolInfo.queueFamilyIndex = queueFamily(
                        queueFamilies, static_cast<QueueType>(queue));

                poolInfo.flags =
                    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

                    if (vkCreateCommandPool(
                            m_device, &poolInfo, nullptr, &worker.pool) !=
                        VK_SUCCESS)
                    {
                        throw std::runtime_error(
                            "Failed to create Vulkan worker command pool.");
                    }
                }
            }
        }

        VkCommandPoolCreateInfo immediatePoolInfo{};
        immediatePoolInfo.sType =
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        immediatePoolInfo.queueFamilyIndex =
            queueFamilies.graphics;
        immediatePoolInfo.flags =
            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        if (vkCreateCommandPool(
            m_device,
            &immediatePoolInfo,
            nullptr,
            &m_immediatePool) != VK_SUCCESS)
        {
            throw std::runtime_error(
                "Failed to create Vulkan immediate command pool.");
        }

        VkCommandBufferAllocateInfo immediateAllocInfo{};
        immediateAllocInfo.sType =
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        immediateAllocInfo.commandPool =
            m_immediatePool;
        immediateAllocInfo.level =
            VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        immediateAllocInfo.commandBufferCount =
            1;

        if (vkAllocateCommandBuffers(
            m_device,
            &immediateAllocInfo,
            &m_immediateCommandBuffer) != VK_SUCCESS)
        {
            throw std::runtime_error(
                "Failed to allocate Vulkan immediate command buffer.");
        }

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType =
            VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

        fenceInfo.flags =
            VK_FENCE_CREATE_SIGNALED_BIT;

        if (vkCreateFence(
            m_device,
            &fenceInfo,
            nullptr,
            &m_immediateFence) != VK_SUCCESS)
        {
            throw std::runtime_error(
                "Failed to create Vulkan immediate fence.");
        }

        spdlog::info("[VulkanCommandSystem] Initialized (frames={}, workers={})",
            maxFramesInFlight,
            maxWorkers);
    }

    void VulkanCommandSystem::shutdown()
    {
        vkDeviceWaitIdle(m_device);

        for (auto& frame : m_frames)
        {
            for (auto& workers : frame.queuePools)
            for (std::unique_ptr<WorkerPool>& workerPtr : workers)
            {
                WorkerPool& worker = *workerPtr;

                if (worker.pool != VK_NULL_HANDLE)
                {
                    vkDestroyCommandPool(
                        m_device,
                        worker.pool,
                        nullptr);
                }
            }
        }

        if (m_immediatePool != VK_NULL_HANDLE)
        {
            vkDestroyCommandPool(
                m_device,
                m_immediatePool,
                nullptr);
            m_immediatePool = VK_NULL_HANDLE;
            m_immediateCommandBuffer = VK_NULL_HANDLE;
        }

        if (m_immediateFence)
        {
            vkDestroyFence(
                m_device,
                m_immediateFence,
                nullptr);
            m_immediateFence = VK_NULL_HANDLE;
        }

        m_frames.clear();
    }

    void VulkanCommandSystem::beginFrame(uint32_t frameIndex)
    {
        if (frameIndex >= m_frames.size())
        {
            throw std::runtime_error("Vulkan frame index out of range.");
        }
        for (auto& workers : m_frames[frameIndex].queuePools)
        for (std::unique_ptr<WorkerPool>& workerPtr : workers)
        {
            WorkerPool& worker = *workerPtr;
            std::scoped_lock lock(worker.mutex);
            worker.nextCommandBuffer = 0;
            if (vkResetCommandPool(
                    m_device, worker.pool, 0) != VK_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to reset Vulkan worker command pool.");
            }
        }
    }

    VulkanCommandSystem::RecordingLease
        VulkanCommandSystem::acquireFrameCommandBuffer(
            uint32_t frameIndex,
            uint32_t workerIndex,
            QueueType queue)
    {
        if (frameIndex >= m_frames.size())
        {
            throw std::runtime_error("Vulkan frame index out of range.");
        }
        FrameData& frame = m_frames[frameIndex];

        auto& workers = frame.queuePools[queueIndex(queue)];
        if (workerIndex >= workers.size())
        {
            throw std::runtime_error("Vulkan worker index out of range.");
        }

        WorkerPool& pool =
            *workers[workerIndex];

        std::unique_lock lock(pool.mutex);

        if (pool.nextCommandBuffer >= pool.commandBuffers.size())
        {
            VkCommandBufferAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.commandPool = pool.pool;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = 1;

            VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

            if (vkAllocateCommandBuffers(
                m_device,
                &allocInfo,
                &commandBuffer) != VK_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to allocate Vulkan frame command buffer.");
            }

            pool.commandBuffers.push_back(commandBuffer);
        }

        VkCommandBuffer commandBuffer =
            pool.commandBuffers[pool.nextCommandBuffer++];

        return RecordingLease(commandBuffer, std::move(lock));
    }

    void VulkanCommandSystem::immediateSubmit(
        VkQueue queue,
        std::function<void(VkCommandBuffer cmd)>&& function)
    {
        std::scoped_lock lock(m_immediateMutex);

        if (m_immediatePool == VK_NULL_HANDLE ||
            m_immediateCommandBuffer == VK_NULL_HANDLE ||
            m_immediateFence == VK_NULL_HANDLE)
        {
            throw std::runtime_error(
                "Vulkan immediate submit used before command system init.");
        }

        if (vkWaitForFences(
            m_device,
            1,
            &m_immediateFence,
            VK_TRUE,
            UINT64_MAX) != VK_SUCCESS)
        {
            throw std::runtime_error(
                "Failed to wait for Vulkan immediate fence.");
        }

        if (vkResetFences(
            m_device,
            1,
            &m_immediateFence) != VK_SUCCESS)
        {
            throw std::runtime_error(
                "Failed to reset Vulkan immediate fence.");
        }

        if (vkResetCommandPool(
            m_device,
            m_immediatePool,
            0) != VK_SUCCESS)
        {
            throw std::runtime_error(
                "Failed to reset Vulkan immediate command pool.");
        }

        VkCommandBuffer cmd = m_immediateCommandBuffer;

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType =
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags =
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS)
        {
            throw std::runtime_error(
                "Failed to begin Vulkan immediate command buffer.");
        }

        function(cmd);

        if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
        {
            throw std::runtime_error(
                "Failed to end Vulkan immediate command buffer.");
        }

        VkSubmitInfo submitInfo{};
        submitInfo.sType =
            VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount =
            1;
        submitInfo.pCommandBuffers =
            &cmd;

        if (vkQueueSubmit(
            queue,
            1,
            &submitInfo,
            m_immediateFence) != VK_SUCCESS)
        {
            throw std::runtime_error(
                "Failed to submit Vulkan immediate command buffer.");
        }

        if (vkWaitForFences(
            m_device,
            1,
            &m_immediateFence,
            VK_TRUE,
            UINT64_MAX) != VK_SUCCESS)
        {
            throw std::runtime_error(
                "Failed to wait for Vulkan immediate submission.");
        }
    }
}
