#pragma once
#include <vulkan/vulkan.h>

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
            uint32_t graphicsQueueFamily,
            uint32_t maxFramesInFlight,
            uint32_t maxWorkers);

        void shutdown();

        void beginFrame(uint32_t frameIndex);

        VkCommandBuffer beginFrameCommandBuffer(
            uint32_t frameIndex,
            uint32_t workerIndex);

        VkCommandBuffer allocateCommandBuffer(
            uint32_t frameIndex,
            uint32_t workerIndex);

        // Immediate submission (upload / init / staging)
        void immediateSubmit(
            VkQueue queue,
            std::function<void(VkCommandBuffer cmd)>&& function);

    private:

        struct WorkerPool
        {
            VkCommandPool pool = VK_NULL_HANDLE;
            VkCommandBuffer primary = VK_NULL_HANDLE;
        };

        struct FrameData
        {
            std::vector<WorkerPool> workerPools;
        };

        VkDevice m_device = VK_NULL_HANDLE;
        uint32_t m_graphicsQueueFamily = 0;

        std::vector<FrameData> m_frames;

        VkFence m_immediateFence = VK_NULL_HANDLE;
    };

}
