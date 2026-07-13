#pragma once
#include <vulkan/vulkan.h>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>
#include <array>
#include "ic/renderer/frame_graph/frame_graph_types.h"
#include "vulkan_common.h"

/*

Each worker thread gets its own command pool.

Because Vulkan command pools are not thread safe.

Then each pass can record independently.

Later the DAG scheduler can do:

Pass A
	Thread 0

Pass B
	Thread 1

Pass C
	Thread 2

Each thread records into its own command buffer.

At the end of the frame:

Submit primary command buffers

*/

namespace ic
{
    class VulkanCommandSystem
    {
    public:

        void init(
            VkDevice device,
            const QueueFamilyIndices& queueFamilies,
            uint32_t maxFramesInFlight,
            uint32_t maxWorkers);

        void shutdown();

        void beginFrame(uint32_t frameIndex);

        class RecordingLease
        {
        public:
            RecordingLease() = default;
            RecordingLease(const RecordingLease&) = delete;
            RecordingLease& operator=(const RecordingLease&) = delete;
            RecordingLease(RecordingLease&&) noexcept = default;
            RecordingLease& operator=(RecordingLease&&) noexcept = default;

            VkCommandBuffer commandBuffer() const
            {
                return m_commandBuffer;
            }

            explicit operator bool() const
            {
                return m_commandBuffer != VK_NULL_HANDLE;
            }

        private:
            friend class VulkanCommandSystem;

            RecordingLease(
                VkCommandBuffer commandBuffer,
                std::unique_lock<std::mutex>&& lock)
                : m_commandBuffer(commandBuffer)
                , m_lock(std::move(lock))
            {
            }

            VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
            std::unique_lock<std::mutex> m_lock;
        };

        RecordingLease acquireFrameCommandBuffer(
            uint32_t frameIndex,
            uint32_t workerIndex,
            QueueType queue = QueueType::Graphics);

        // Immediate submission (upload / init / staging)
        void immediateSubmit(
            VkQueue queue,
            std::function<void(VkCommandBuffer cmd)>&& function);

    private:

        struct WorkerPool
        {
            VkCommandPool pool = VK_NULL_HANDLE;
            std::vector<VkCommandBuffer> commandBuffers;
            uint32_t nextCommandBuffer = 0;
            std::mutex mutex;
        };

        struct FrameData
        {
            std::array<
                std::vector<std::unique_ptr<WorkerPool>>, 3> queuePools;
        };

        VkDevice m_device = VK_NULL_HANDLE;
        QueueFamilyIndices m_queueFamilies{};

        std::vector<FrameData> m_frames;

        VkCommandPool m_immediatePool = VK_NULL_HANDLE;
        VkCommandBuffer m_immediateCommandBuffer = VK_NULL_HANDLE;
        VkFence m_immediateFence = VK_NULL_HANDLE;
        std::mutex m_immediateMutex;
    };

}
