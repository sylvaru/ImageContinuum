#include "ic/renderer/dx12_backend/dx12_frame_executor.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <spdlog/spdlog.h>

namespace ic
{
    namespace
    {
        void throwIfFailed(HRESULT hr, const char* message)
        {
            if (FAILED(hr))
            {
                throw std::runtime_error(
                    std::string(message) + " (HRESULT=" +
                    std::to_string(static_cast<uint32_t>(hr)) + ")");
            }
        }
    }

    void DX12FrameExecutor::init(
        const DX12Device& device,
        DX12Swapchain& swapchain,
        uint32_t framesInFlight)
    {
        m_device = &device;
        m_swapchain = &swapchain;
        m_frameSync.assign(std::max(1u, framesInFlight), FrameSync{});

        throwIfFailed(
            device.device()->CreateFence(
                0,
                D3D12_FENCE_FLAG_NONE,
                IID_PPV_ARGS(&m_fence)),
            "Failed to create DX12 frame fence.");

        for (auto& queueFence : m_queueFences)
        {
            throwIfFailed(
                device.device()->CreateFence(
                    0, D3D12_FENCE_FLAG_NONE,
                    IID_PPV_ARGS(&queueFence)),
                "Failed to create DX12 queue fence.");
        }

        m_fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!m_fenceEvent)
        {
            throw std::runtime_error("Failed to create DX12 fence event.");
        }

        m_nextFenceValue = 1;
        m_nextQueueFenceValues = { 1, 1, 1 };
        m_lastGraphCompletionValue = 0;
    }

    void DX12FrameExecutor::shutdown()
    {
        if (m_fenceEvent)
        {
            CloseHandle(m_fenceEvent);
            m_fenceEvent = nullptr;
        }

        m_fence.Reset();
        for (auto& queueFence : m_queueFences)
        {
            queueFence.Reset();
        }
        m_frameSync.clear();
        m_nextFenceValue = 1;
        m_nextQueueFenceValues = { 1, 1, 1 };
        m_lastGraphCompletionValue = 0;
        m_device = nullptr;
        m_swapchain = nullptr;
    }

    void DX12FrameExecutor::waitForFrame(uint32_t frameSlot)
    {
        if (!m_fence || frameSlot >= m_frameSync.size())
        {
            return;
        }

        const uint64_t fenceValue = m_frameSync[frameSlot].fenceValue;

        if (fenceValue == 0 || m_fence->GetCompletedValue() >= fenceValue)
        {
            return;
        }

        throwIfFailed(
            m_fence->SetEventOnCompletion(fenceValue, m_fenceEvent),
            "Failed to wait on DX12 frame fence.");

        WaitForSingleObject(m_fenceEvent, INFINITE);
    }

    void DX12FrameExecutor::signalFrame(uint32_t frameSlot)
    {
        const uint64_t fenceValue = m_nextFenceValue++;

        throwIfFailed(
            m_device->graphicsQueue()->Signal(m_fence.Get(), fenceValue),
            "Failed to signal DX12 frame fence.");

        m_frameSync[frameSlot].fenceValue = fenceValue;
        m_lastGraphCompletionValue = fenceValue;
    }

    void DX12FrameExecutor::waitForGpu()
    {
        if (!m_device || !m_device->graphicsQueue() || !m_fence)
        {
            return;
        }

        const uint64_t fenceValue = m_nextFenceValue++;

        throwIfFailed(
            m_device->graphicsQueue()->Signal(m_fence.Get(), fenceValue),
            "Failed to signal DX12 idle fence.");

        if (m_fence->GetCompletedValue() < fenceValue)
        {
            throwIfFailed(
                m_fence->SetEventOnCompletion(fenceValue, m_fenceEvent),
                "Failed to wait for DX12 idle fence.");

            WaitForSingleObject(m_fenceEvent, INFINITE);
        }

        for (FrameSync& sync : m_frameSync)
        {
            sync.fenceValue = 0;
        }
    }

    bool DX12FrameExecutor::submitAndPresent(
        const CompiledGraphPlan& plan,
        const std::vector<ID3D12CommandList*>& commandLists,
        uint32_t frameSlot,
        const DX12UploadDependency& uploadDependency)
    {
        std::array<bool, 3> uploadDependencyApplied{};
        const auto queueIndex = [](QueueType queue)
        {
            return static_cast<uint32_t>(queue);
        };
        auto waitForUploads =
            [&](ID3D12CommandQueue* queue, QueueType queueType)
            {
                const uint32_t index = queueIndex(queueType);
                if (!uploadDependency.valid() ||
                    uploadDependencyApplied[index])
                {
                    return;
                }
                // Submission order is already sufficient on the upload queue.
                // All other queues require an explicit GPU-side fence wait.
                if (queue != uploadDependency.queue)
                {
                    throwIfFailed(
                        queue->Wait(
                            uploadDependency.fence,
                            uploadDependency.value),
                        "Failed to enqueue DX12 upload dependency.");
                }
                uploadDependencyApplied[index] = true;
            };

        if (!plan.queueSubmissions.empty())
        {
            static bool loggedSchedule = false;
            if (!loggedSchedule)
            {
                for (uint32_t i = 0; i < plan.queueSubmissions.size(); ++i)
                {
                    const QueueSubmissionBatch& batch = plan.queueSubmissions[i];
                    spdlog::debug(
                        "[DX12FrameExecutor] batch={} level={} queue={} nodes={} waits={}",
                        i, batch.levelIndex, static_cast<uint32_t>(batch.queue),
                        batch.nodeCount, batch.waitCount);
                }
                loggedSchedule = true;
            }
            std::vector<uint32_t> commandIndexByNode(
                plan.nodes.size(), UINT32_MAX);
            for (uint32_t i = 0;
                 i < plan.executionLevelNodes.size(); ++i)
            {
                commandIndexByNode[plan.executionLevelNodes[i]] = i;
            }

            struct SubmissionSignal
            {
                QueueType queue = QueueType::Graphics;
                uint64_t value = 0;
            };
            std::vector<SubmissionSignal> submissionSignals(
                plan.queueSubmissions.size());
            std::array<uint64_t, 3> lastQueueSignals{};

            auto queueFor = [this](QueueType queue)
            {
                switch (queue)
                {
                case QueueType::Compute: return m_device->computeQueue();
                case QueueType::Transfer: return m_device->copyQueue();
                case QueueType::Graphics: return m_device->graphicsQueue();
                }
                return m_device->graphicsQueue();
            };

            for (uint32_t submissionIndex = 0;
                 submissionIndex < plan.queueSubmissions.size();
                 ++submissionIndex)
            {
                const QueueSubmissionBatch& submission =
                    plan.queueSubmissions[submissionIndex];
                ID3D12CommandQueue* queue = queueFor(submission.queue);
                waitForUploads(queue, submission.queue);

                if (submission.levelIndex == 0 &&
                    m_lastGraphCompletionValue != 0)
                {
                    throwIfFailed(
                        queue->Wait(
                            m_fence.Get(), m_lastGraphCompletionValue),
                        "Failed to wait for prior DX12 graph frame.");
                }

                for (uint32_t i = 0; i < submission.waitCount; ++i)
                {
                    const uint32_t dependency =
                        plan.queueSubmissionWaits[
                            submission.firstWait + i].submissionIndex;
                    if (dependency < submissionSignals.size() &&
                        submissionSignals[dependency].value != 0)
                    {
                        const SubmissionSignal& source =
                            submissionSignals[dependency];
                        if (queueFor(source.queue) == queue)
                        {
                            continue;
                        }
                        throwIfFailed(
                            queue->Wait(
                                m_queueFences[queueIndex(source.queue)].Get(),
                                source.value),
                            "Failed to enqueue DX12 cross-queue wait.");
                    }
                }

                std::vector<ID3D12CommandList*> batchLists;
                batchLists.reserve(submission.nodeCount);
                for (uint32_t i = 0; i < submission.nodeCount; ++i)
                {
                    const GraphNodeId node =
                        plan.queueSubmissionNodes[submission.firstNode + i];
                    const uint32_t commandIndex = commandIndexByNode[node];
                    if (commandIndex != UINT32_MAX &&
                        commandIndex < commandLists.size())
                    {
                        batchLists.push_back(commandLists[commandIndex]);
                    }
                }
                if (!batchLists.empty())
                {
                    queue->ExecuteCommandLists(
                        static_cast<UINT>(batchLists.size()),
                        batchLists.data());
                }

                const uint32_t signalQueue = queueIndex(submission.queue);
                const uint64_t signal =
                    m_nextQueueFenceValues[signalQueue]++;
                throwIfFailed(
                    queue->Signal(m_queueFences[signalQueue].Get(), signal),
                    "Failed to signal DX12 queue submission.");
                submissionSignals[submissionIndex] = {
                    submission.queue, signal };
                lastQueueSignals[signalQueue] = signal;
            }

            const uint32_t computeIndex = queueIndex(QueueType::Compute);
            if (lastQueueSignals[computeIndex] != 0)
            {
                throwIfFailed(
                    m_device->graphicsQueue()->Wait(
                        m_queueFences[computeIndex].Get(),
                        lastQueueSignals[computeIndex]),
                    "Failed to join DX12 compute queue.");
            }
            const uint32_t copyIndex = queueIndex(QueueType::Transfer);
            if (lastQueueSignals[copyIndex] != 0)
            {
                throwIfFailed(
                    m_device->graphicsQueue()->Wait(
                        m_queueFences[copyIndex].Get(),
                        lastQueueSignals[copyIndex]),
                    "Failed to join DX12 copy queue.");
            }

            const size_t graphCommandCount =
                plan.executionLevelNodes.size();
            if (commandLists.size() > graphCommandCount)
            {
                waitForUploads(
                    m_device->graphicsQueue(), QueueType::Graphics);
                m_device->graphicsQueue()->ExecuteCommandLists(
                    static_cast<UINT>(
                        commandLists.size() - graphCommandCount),
                    commandLists.data() + graphCommandCount);
            }
        }
        else if (!commandLists.empty())
        {
            waitForUploads(
                m_device->graphicsQueue(), QueueType::Graphics);
            m_device->graphicsQueue()->ExecuteCommandLists(
                static_cast<UINT>(commandLists.size()),
                commandLists.data());
        }

        // The graphics frame fence owns slot reuse and retirement. Join the
        // upload fence even when this frame had no graphics batch so its signal
        // cannot pass pending copy work.
        waitForUploads(m_device->graphicsQueue(), QueueType::Graphics);

        const bool presented = m_swapchain->present();
        signalFrame(frameSlot);
        return presented;
    }
}
