#pragma once

#include "ic/renderer/vulkan_backend/vulkan_device.h"
#include "ic/renderer/vulkan_backend/vulkan_swapchain.h"
#include "ic/renderer/frame_graph/compiled_graph_plan.h"
#include "ic/renderer/vulkan_backend/vulkan_upload_scheduler.h"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include <vulkan/vulkan.h>

namespace ic
{
    // Owns per frame and per swapchain image synchronization (in flight fences,
    // image available / render finished semaphores, per queue timeline
    // semaphores) and drives multi queue submission and presentation for a
    // recorded frame. The backend records the command buffers (via the
    // frame graph parallel recorder) and delegates acquire, submission, fences
    // and present here.
    class VulkanFrameExecutor final
    {
    public:
        struct AcquiredFrame
        {
            uint32_t imageIndex = UINT32_MAX;
            VkImage swapchainImage = VK_NULL_HANDLE;
            VkImageLayout initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            bool acquired = false;
        };

        VulkanFrameExecutor() = default;
        ~VulkanFrameExecutor() noexcept
        {
            shutdown();
        }

        VulkanFrameExecutor(const VulkanFrameExecutor&) = delete;
        VulkanFrameExecutor& operator=(const VulkanFrameExecutor&) = delete;

        VulkanFrameExecutor(VulkanFrameExecutor&&) = delete;
        VulkanFrameExecutor& operator=(VulkanFrameExecutor&&) = delete;

        void init(
            const VulkanDevice& device,
            VulkanSwapchain& swapchain,
            uint32_t framesInFlight);
        void shutdown();

        // Recreates the per swapchain image sync objects (after a resize).
        void initSwapchainSync();

        [[nodiscard]] uint32_t framesInFlight() const noexcept
        {
            return static_cast<uint32_t>(m_frameSync.size());
        }

        [[nodiscard]] bool ready() const noexcept
        {
            return !m_frameSync.empty();
        }

        [[nodiscard]] uint32_t currentSwapchainImage() const noexcept
        {
            return m_currentSwapchainImage;
        }

        // Blocks until the GPU work previously submitted for this slot has
        // completed.
        void waitForFrameSlot(uint32_t frameSlot);

        // Acquires the next swapchain image and prepares this frame's fence.
        // Returns acquired == false when the swapchain must be recreated.
        AcquiredFrame acquire(uint32_t frameSlot);

        // Submits the recorded command buffers honoring the compiled queue
        // schedule and timeline semaphores, transitions the presented image
        // layout, and presents. Returns whether present succeeded.
        bool submitAndPresent(
            const CompiledGraphPlan& plan,
            std::span<const VkCommandBuffer> commandBuffers,
            uint32_t frameSlot,
            const GraphExecutionContext& execution,
            VulkanUploadDependency uploadDependency = {});

    private:
        struct FrameSync
        {
            VkFence inFlightFence = VK_NULL_HANDLE;
            VkSemaphore imageAvailable = VK_NULL_HANDLE;
        };

        // The producing queue + timeline value each of the PREVIOUS frame's
        // submissions signaled, indexed by submission index. Cross-frame ordering
        // edges make this frame's consuming submission wait on the producer's
        // prior-frame value. This is the minimal replacement for the old blanket
        // previous-frame barrier. Reset on a full GPU drain / graph rebuild.
        struct CrossFrameSignal
        {
            QueueType queue = QueueType::Graphics;
            uint64_t value = 0;
        };
        std::vector<CrossFrameSignal> m_prevSubmissionSignals;
        std::vector<std::vector<CrossFrameSignal>> m_crossFrameWaits;
        std::vector<uint32_t> m_commandIndexByNode;
        std::vector<CrossFrameSignal> m_submissionSignals;
        std::vector<VkQueue> m_uploadWaitedQueues;
        std::vector<VkSemaphore> m_waitSemaphores;
        std::vector<uint64_t> m_waitValues;
        std::vector<VkPipelineStageFlags> m_waitStages;
        std::vector<VkCommandBuffer> m_batchCommands;

    public:
        // Call only after vkDeviceWaitIdle. Submission indices from the prior
        // compiled plan are then stale and must not be carried into a rebuild.
        void resetAfterGpuDrain() { m_prevSubmissionSignals.clear(); }

    private:

        void destroyFrameSync();
        void destroySwapchainSync();

        const VulkanDevice* m_device = nullptr;
        VulkanSwapchain* m_swapchain = nullptr;

        std::vector<FrameSync> m_frameSync;
        std::vector<VkSemaphore> m_imageRenderFinished;
        std::vector<VkFence> m_imagesInFlight;
        std::vector<VkImageLayout> m_swapchainImageLayouts;
        std::array<VkSemaphore, 3> m_graphTimelines{};
        std::array<uint64_t, 3> m_nextGraphTimelineValues{ 1, 1, 1 };
        uint32_t m_currentSwapchainImage = 0;
    };
}
