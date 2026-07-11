#include "ic/renderer/dx12_backend/dx12_frame_executor.h"

#include <algorithm>
#include <stdexcept>
#include <string>

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

        m_fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!m_fenceEvent)
        {
            throw std::runtime_error("Failed to create DX12 fence event.");
        }

        m_nextFenceValue = 1;
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
        m_frameSync.clear();
        m_nextFenceValue = 1;
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
        uint32_t frameSlot)
    {
        if (!plan.queueSubmissions.empty())
        {
            std::vector<uint32_t> commandIndexByNode(
                plan.nodes.size(), UINT32_MAX);
            for (uint32_t i = 0;
                 i < plan.executionLevelNodes.size(); ++i)
            {
                commandIndexByNode[plan.executionLevelNodes[i]] = i;
            }

            std::vector<uint64_t> submissionSignals(
                plan.queueSubmissions.size(), 0);
            uint64_t lastComputeSignal = 0;
            uint64_t lastCopySignal = 0;

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
                        submissionSignals[dependency] != 0)
                    {
                        throwIfFailed(
                            queue->Wait(
                                m_fence.Get(),
                                submissionSignals[dependency]),
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

                const uint64_t signal = m_nextFenceValue++;
                throwIfFailed(
                    queue->Signal(m_fence.Get(), signal),
                    "Failed to signal DX12 queue submission.");
                submissionSignals[submissionIndex] = signal;
                if (submission.queue == QueueType::Compute)
                {
                    lastComputeSignal = signal;
                }
                else if (submission.queue == QueueType::Transfer)
                {
                    lastCopySignal = signal;
                }
            }

            if (lastComputeSignal != 0)
            {
                throwIfFailed(
                    m_device->graphicsQueue()->Wait(
                        m_fence.Get(), lastComputeSignal),
                    "Failed to join DX12 compute queue.");
            }
            if (lastCopySignal != 0)
            {
                throwIfFailed(
                    m_device->graphicsQueue()->Wait(
                        m_fence.Get(), lastCopySignal),
                    "Failed to join DX12 copy queue.");
            }

            const size_t graphCommandCount =
                plan.executionLevelNodes.size();
            if (commandLists.size() > graphCommandCount)
            {
                m_device->graphicsQueue()->ExecuteCommandLists(
                    static_cast<UINT>(
                        commandLists.size() - graphCommandCount),
                    commandLists.data() + graphCommandCount);
            }
        }
        else if (!commandLists.empty())
        {
            m_device->graphicsQueue()->ExecuteCommandLists(
                static_cast<UINT>(commandLists.size()),
                commandLists.data());
        }

        const bool presented = m_swapchain->present();
        signalFrame(frameSlot);
        return presented;
    }
}
