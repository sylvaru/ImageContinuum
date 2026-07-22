#include "ic/renderer/vulkan_backend/vulkan_frame_executor.h"

#include <spdlog/spdlog.h>

#include <stdexcept>
#include <string>

namespace ic
{
    namespace
    {
        void throwIfFailed(VkResult result, const char* message)
        {
            if (result != VK_SUCCESS)
            {
                throw std::runtime_error(
                    std::string(message) + " VkResult=" +
                    std::to_string(static_cast<int32_t>(result)));
            }
        }
    }

    void VulkanFrameExecutor::init(
        const VulkanDevice& device,
        VulkanSwapchain& swapchain,
        uint32_t framesInFlight)
    {
        m_device = &device;
        m_swapchain = &swapchain;

        const uint32_t frames = framesInFlight == 0 ? 1 : framesInFlight;
        m_frameSync.resize(frames);

        for (auto& fs : m_frameSync)
        {
            VkSemaphoreCreateInfo semInfo{
                VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
            if (vkCreateSemaphore(
                    device.device(),
                    &semInfo,
                    nullptr,
                    &fs.imageAvailable) != VK_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to create Vulkan image-available semaphore.");
            }

            VkFenceCreateInfo fenceInfo{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
            fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            if (vkCreateFence(
                    device.device(),
                    &fenceInfo,
                    nullptr,
                    &fs.inFlightFence) != VK_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to create Vulkan frame fence.");
            }
        }

        VkSemaphoreTypeCreateInfo timelineType{};
        timelineType.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        timelineType.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        timelineType.initialValue = 0;
        VkSemaphoreCreateInfo timelineInfo{};
        timelineInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        timelineInfo.pNext = &timelineType;
        for (VkSemaphore& timeline : m_graphTimelines)
        {
            throwIfFailed(
                vkCreateSemaphore(
                    device.device(), &timelineInfo, nullptr, &timeline),
                "Failed to create Vulkan graph timeline semaphore.");
        }

        initSwapchainSync();
    }

    void VulkanFrameExecutor::initSwapchainSync()
    {
        destroySwapchainSync();

        // A swapchain recreate drains the GPU and rebuilds the graph; drop the
        // previous frame's submission signals so no cross-frame edge waits on a
        // value from a plan that no longer exists.
        m_prevSubmissionSignals.clear();

        const size_t imageCount = m_swapchain->images().size();

        m_imageRenderFinished.resize(imageCount, VK_NULL_HANDLE);
        m_imagesInFlight.assign(imageCount, VK_NULL_HANDLE);
        m_swapchainImageLayouts.assign(imageCount, VK_IMAGE_LAYOUT_UNDEFINED);

        VkSemaphoreCreateInfo semInfo{
            VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        for (VkSemaphore& semaphore : m_imageRenderFinished)
        {
            if (vkCreateSemaphore(
                    m_device->device(),
                    &semInfo,
                    nullptr,
                    &semaphore) != VK_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to create Vulkan render-finished semaphore.");
            }
        }
    }

    void VulkanFrameExecutor::shutdown()
    {
        destroySwapchainSync();
        destroyFrameSync();

        if (m_device && m_device->device() != VK_NULL_HANDLE)
        {
            for (VkSemaphore& timeline : m_graphTimelines)
            {
                if (timeline != VK_NULL_HANDLE)
                {
                    vkDestroySemaphore(m_device->device(), timeline, nullptr);
                    timeline = VK_NULL_HANDLE;
                }
            }
        }

        m_nextGraphTimelineValues = { 1, 1, 1 };
        m_prevSubmissionSignals.clear();
        m_currentSwapchainImage = 0;
        m_device = nullptr;
        m_swapchain = nullptr;
    }

    void VulkanFrameExecutor::destroyFrameSync()
    {
        if (!m_device || m_device->device() == VK_NULL_HANDLE)
        {
            return;
        }
        VkDevice device = m_device->device();

        for (FrameSync& fs : m_frameSync)
        {
            if (fs.imageAvailable != VK_NULL_HANDLE)
            {
                vkDestroySemaphore(device, fs.imageAvailable, nullptr);
                fs.imageAvailable = VK_NULL_HANDLE;
            }
            if (fs.inFlightFence != VK_NULL_HANDLE)
            {
                vkDestroyFence(device, fs.inFlightFence, nullptr);
                fs.inFlightFence = VK_NULL_HANDLE;
            }
        }

        m_frameSync.clear();
    }

    void VulkanFrameExecutor::destroySwapchainSync()
    {
        if (!m_device || m_device->device() == VK_NULL_HANDLE)
        {
            return;
        }
        VkDevice device = m_device->device();

        for (VkSemaphore semaphore : m_imageRenderFinished)
        {
            if (semaphore != VK_NULL_HANDLE)
            {
                vkDestroySemaphore(device, semaphore, nullptr);
            }
        }

        m_imageRenderFinished.clear();
        m_imagesInFlight.clear();
        m_swapchainImageLayouts.clear();
    }

    void VulkanFrameExecutor::waitForFrameSlot(uint32_t frameSlot)
    {
        if (frameSlot >= m_frameSync.size())
        {
            return;
        }
        throwIfFailed(
            vkWaitForFences(
                m_device->device(), 1,
                &m_frameSync[frameSlot].inFlightFence,
                VK_TRUE, UINT64_MAX),
            "Failed to wait for Vulkan frame-slot fence.");
    }

    VulkanFrameExecutor::AcquiredFrame VulkanFrameExecutor::acquire(
        uint32_t frameSlot)
    {
        AcquiredFrame frame{};

        FrameSync& sync = m_frameSync[frameSlot];

        const uint32_t imageIndex =
            m_swapchain->acquireNextImage(sync.imageAvailable);
        if (imageIndex == UINT32_MAX)
        {
            return frame;
        }

        m_currentSwapchainImage = imageIndex;

        frame.imageIndex = imageIndex;
        frame.swapchainImage = m_swapchain->image(imageIndex);
        frame.initialLayout =
            imageIndex < m_swapchainImageLayouts.size()
                ? m_swapchainImageLayouts[imageIndex]
                : VK_IMAGE_LAYOUT_UNDEFINED;
        frame.acquired = true;

        if (m_imagesInFlight[imageIndex] != VK_NULL_HANDLE)
        {
            throwIfFailed(
                vkWaitForFences(
                    m_device->device(), 1,
                    &m_imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX),
                "Failed to wait for Vulkan swapchain-image fence.");
        }

        m_imagesInFlight[imageIndex] = sync.inFlightFence;

        throwIfFailed(
            vkResetFences(m_device->device(), 1, &sync.inFlightFence),
            "Failed to reset Vulkan frame-slot fence.");

        return frame;
    }

    bool VulkanFrameExecutor::submitAndPresent(
        const CompiledGraphPlan& plan,
        std::span<const VkCommandBuffer> commandBuffers,
        uint32_t frameSlot,
        const GraphExecutionContext& execution,
        VulkanUploadDependency uploadDependency)
    {
        FrameSync& sync = m_frameSync[frameSlot];
        const uint32_t imageIndex = m_currentSwapchainImage;
        VkSemaphore renderFinished = m_imageRenderFinished[imageIndex];

        auto queueIndex = [](QueueType queue)
        {
            return static_cast<uint32_t>(queue);
        };

        // Translate the compiler's cross-frame edges into per-submission waits on
        // the PREVIOUS frame's producing submission (see the DX12 executor for the
        // shared rationale). Only edges whose consumer node executes this frame
        // are applied, and only while the previous-frame signal map still matches
        // this plan (dropped on a GPU drain / graph rebuild).
        m_crossFrameWaits.resize(plan.queueSubmissions.size());
        for (auto& waits : m_crossFrameWaits)
        {
            waits.clear();
        }
        const bool prevSignalsUsable =
            m_prevSubmissionSignals.size() == plan.queueSubmissions.size();
        if (prevSignalsUsable)
        {
            for (const CrossFrameDependency& dep : plan.crossFrameDependencies)
            {
                if (!execution.shouldExecute(dep.consumerNode) ||
                    dep.consumerSubmission >= m_crossFrameWaits.size() ||
                    dep.producerSubmission >= m_prevSubmissionSignals.size())
                {
                    continue;
                }
                const CrossFrameSignal& producer =
                    m_prevSubmissionSignals[dep.producerSubmission];
                if (producer.value != 0)
                {
                    m_crossFrameWaits[dep.consumerSubmission].push_back(producer);
                }
            }
        }

        static bool loggedCrossFrame = false;
        if (!loggedCrossFrame)
        {
            spdlog::info(
                "[VulkanFrameExecutor] cross-frame deps={} "
                "(0 => consecutive frames overlap fully; the old blanket "
                "previous-frame barrier is removed)",
                plan.crossFrameDependencies.size());
            loggedCrossFrame = true;
        }
        auto queueFor = [this](QueueType queue)
        {
            switch (queue)
            {
            case QueueType::Compute: return m_device->computeQueue();
            case QueueType::Transfer: return m_device->transferQueue();
            case QueueType::Graphics: return m_device->graphicsQueue();
            }
            return m_device->graphicsQueue();
        };

        m_commandIndexByNode.assign(plan.nodes.size(), UINT32_MAX);
        for (uint32_t i = 0; i < plan.executionLevelNodes.size(); ++i)
        {
            m_commandIndexByNode[plan.executionLevelNodes[i]] = i;
        }

        m_submissionSignals.assign(
            plan.queueSubmissions.size(), CrossFrameSignal{});
        std::array<uint64_t, 3> lastQueueSignals{};
        bool imageAvailableConsumed = false;
        m_uploadWaitedQueues.clear();

        const auto batchTouchesSwapchain = [&plan](
            const QueueSubmissionBatch& batch)
        {
            for (uint32_t i = 0; i < batch.nodeCount; ++i)
            {
                const GraphNodeId nodeId =
                    plan.queueSubmissionNodes[batch.firstNode + i];
                const ExecutionNode& node = plan.nodes[nodeId];
                const auto accesses = plan.resourceAccesses.subspan(
                    node.firstResourceAccess, node.resourceAccessCount);
                for (const ResourceAccess& access : accesses)
                {
                    if (access.resource < plan.resources.size())
                    {
                        const GraphResource& resource =
                            plan.resources[access.resource];
                        if (resource.ownership == ResourceOwnership::Imported &&
                            resource.imported == ImportedResource::Swapchain)
                        {
                            return true;
                        }
                    }
                }
            }
            return false;
        };

        for (uint32_t submissionIndex = 0;
             submissionIndex < plan.queueSubmissions.size();
             ++submissionIndex)
        {
            const QueueSubmissionBatch& batch =
                plan.queueSubmissions[submissionIndex];
            const uint32_t destinationQueue = queueIndex(batch.queue);

            m_waitSemaphores.clear();
            m_waitValues.clear();
            m_waitStages.clear();
            const VkQueue batchQueue = queueFor(batch.queue);
            // Per-frame uploads (model geometry/textures) are consumed only by
            // graphics scene-geometry draw passes, so only the graphics queue
            // synchronizes with the upload. Making the async compute/transfer
            // queue wait on it is unnecessary (no such pass reads uploaded data)
            // and needlessly serializes the async work behind the upload; on
            // D3D12 the analogous cross-queue wait also caused device removal.
            // See DX12FrameExecutor::submitAndPresent for the full rationale.
            if (batch.queue == QueueType::Graphics &&
                uploadDependency.value != 0 &&
                batchQueue != uploadDependency.queue &&
                std::ranges::find(m_uploadWaitedQueues, batchQueue) ==
                    m_uploadWaitedQueues.end())
            {
                m_waitSemaphores.push_back(uploadDependency.timeline);
                m_waitValues.push_back(uploadDependency.value);
                m_waitStages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
                m_uploadWaitedQueues.push_back(batchQueue);
            }
            for (uint32_t i = 0; i < batch.waitCount; ++i)
            {
                const uint32_t dependency =
                    plan.queueSubmissionWaits[
                        batch.firstWait + i].submissionIndex;
                const CrossFrameSignal& source =
                    m_submissionSignals[dependency];
                if (queueFor(source.queue) == queueFor(batch.queue))
                {
                    // Submissions to the same physical queue are already
                    // ordered; a timeline wait only adds driver overhead.
                    continue;
                }
                m_waitSemaphores.push_back(
                    m_graphTimelines[queueIndex(source.queue)]);
                m_waitValues.push_back(source.value);
                m_waitStages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
            }
            // Minimal cross-frame ordering: this submission waits only for the
            // specific prior-frame submissions that last used a resource it now
            // reuses (persistent writes, history previous-reads). Per-frame-slot
            // resources are absent from this list, so consecutive frames overlap
                // freely. The blanket "level-0 waits for the whole previous frame"
            // rule is gone. Cross-frame hazards carry no intra-frame barrier, so
            // even a same-queue timeline wait is meaningful here.
            for (const CrossFrameSignal& wait :
                 m_crossFrameWaits[submissionIndex])
            {
                m_waitSemaphores.push_back(
                    m_graphTimelines[queueIndex(wait.queue)]);
                m_waitValues.push_back(wait.value);
                m_waitStages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
            }
            // Do not stall unrelated compute/transfer work on image acquire.
            // Consume the binary semaphore at the first batch that actually
            // accesses the imported swapchain image.
            if (!imageAvailableConsumed && batchTouchesSwapchain(batch))
            {
                m_waitSemaphores.push_back(sync.imageAvailable);
                m_waitValues.push_back(0);
                m_waitStages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
                imageAvailableConsumed = true;
            }

            m_batchCommands.clear();
            m_batchCommands.reserve(batch.nodeCount);
            for (uint32_t i = 0; i < batch.nodeCount; ++i)
            {
                const GraphNodeId node =
                    plan.queueSubmissionNodes[batch.firstNode + i];
                const uint32_t commandIndex = m_commandIndexByNode[node];
                if (commandIndex != UINT32_MAX &&
                    commandIndex < commandBuffers.size())
                {
                    m_batchCommands.push_back(commandBuffers[commandIndex]);
                }
            }

            const uint64_t signalValue =
                m_nextGraphTimelineValues[destinationQueue]++;
            const VkSemaphore signalSemaphore =
                m_graphTimelines[destinationQueue];

            VkTimelineSemaphoreSubmitInfo timeline{};
            timeline.sType =
                VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
            timeline.waitSemaphoreValueCount =
                static_cast<uint32_t>(m_waitValues.size());
            timeline.pWaitSemaphoreValues = m_waitValues.data();
            timeline.signalSemaphoreValueCount = 1;
            timeline.pSignalSemaphoreValues = &signalValue;

            VkSubmitInfo submit{};
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.pNext = &timeline;
            submit.waitSemaphoreCount =
                static_cast<uint32_t>(m_waitSemaphores.size());
            submit.pWaitSemaphores = m_waitSemaphores.data();
            submit.pWaitDstStageMask = m_waitStages.data();
            submit.commandBufferCount =
                static_cast<uint32_t>(m_batchCommands.size());
            submit.pCommandBuffers = m_batchCommands.data();
            submit.signalSemaphoreCount = 1;
            submit.pSignalSemaphores = &signalSemaphore;

            const VkResult submitResult = vkQueueSubmit(
                queueFor(batch.queue), 1, &submit, VK_NULL_HANDLE);
            if (submitResult != VK_SUCCESS)
            {
                spdlog::error(
                    "[VulkanFrameExecutor] submit failed index={} level={} queue={} nodes={} VkResult={}",
                    submissionIndex, batch.levelIndex,
                    static_cast<uint32_t>(batch.queue), batch.nodeCount,
                    static_cast<int32_t>(submitResult));
            }
            throwIfFailed(submitResult,
                "Failed to submit Vulkan frame-graph queue batch.");

            m_submissionSignals[submissionIndex] = {
                batch.queue, signalValue };
            lastQueueSignals[destinationQueue] = signalValue;
        }

        // Publish this frame's per-submission timeline values so the next frame's
        // cross-frame edges can wait on the exact producing submission.
        m_prevSubmissionSignals.assign(
            plan.queueSubmissions.size(), CrossFrameSignal{});
        for (uint32_t i = 0; i < m_submissionSignals.size(); ++i)
        {
            m_prevSubmissionSignals[i] = {
                m_submissionSignals[i].queue, m_submissionSignals[i].value };
        }

        m_waitSemaphores.clear();
        m_waitValues.clear();
        m_waitStages.clear();
        if (uploadDependency.value != 0 &&
            uploadDependency.queue != m_device->graphicsQueue() &&
            std::ranges::find(
                m_uploadWaitedQueues, m_device->graphicsQueue()) ==
                m_uploadWaitedQueues.end())
        {
            m_waitSemaphores.push_back(uploadDependency.timeline);
            m_waitValues.push_back(uploadDependency.value);
            m_waitStages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        }
        for (uint32_t queue = 0; queue < lastQueueSignals.size(); ++queue)
        {
            if (lastQueueSignals[queue] != 0 &&
                queueFor(static_cast<QueueType>(queue)) !=
                    m_device->graphicsQueue())
            {
                m_waitSemaphores.push_back(m_graphTimelines[queue]);
                m_waitValues.push_back(lastQueueSignals[queue]);
                m_waitStages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
            }
        }
        if (!imageAvailableConsumed)
        {
            m_waitSemaphores.push_back(sync.imageAvailable);
            m_waitValues.push_back(0);
            m_waitStages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        }

        const size_t graphCommandCount = plan.executionLevelNodes.size();
        const auto extraCommands = graphCommandCount < commandBuffers.size()
            ? commandBuffers.subspan(graphCommandCount)
            : std::span<const VkCommandBuffer>{};

        VkTimelineSemaphoreSubmitInfo finalTimeline{};
        finalTimeline.sType =
            VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        finalTimeline.waitSemaphoreValueCount =
            static_cast<uint32_t>(m_waitValues.size());
        finalTimeline.pWaitSemaphoreValues = m_waitValues.data();
        const uint32_t graphicsQueueIndex =
            static_cast<uint32_t>(QueueType::Graphics);
        const uint64_t completionValue =
            m_nextGraphTimelineValues[graphicsQueueIndex]++;
        const uint64_t finalSignalValues[] = { completionValue, 0 };
        const VkSemaphore finalSignalSemaphores[] =
        {
            m_graphTimelines[graphicsQueueIndex],
            renderFinished
        };
        finalTimeline.signalSemaphoreValueCount = 2;
        finalTimeline.pSignalSemaphoreValues = finalSignalValues;

        VkSubmitInfo finalSubmit{};
        finalSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        finalSubmit.pNext = &finalTimeline;
        finalSubmit.waitSemaphoreCount =
            static_cast<uint32_t>(m_waitSemaphores.size());
        finalSubmit.pWaitSemaphores = m_waitSemaphores.data();
        finalSubmit.pWaitDstStageMask = m_waitStages.data();
        finalSubmit.commandBufferCount =
            static_cast<uint32_t>(extraCommands.size());
        finalSubmit.pCommandBuffers = extraCommands.data();
        finalSubmit.signalSemaphoreCount = 2;
        finalSubmit.pSignalSemaphores = finalSignalSemaphores;

        throwIfFailed(
            vkQueueSubmit(
                m_device->graphicsQueue(), 1, &finalSubmit, sync.inFlightFence),
            "Failed to join Vulkan graph queues.");

        if (imageIndex < m_swapchainImageLayouts.size())
        {
            m_swapchainImageLayouts[imageIndex] =
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        }

        return m_swapchain->present(
            m_device->presentQueue(),
            imageIndex,
            renderFinished);
    }
}
