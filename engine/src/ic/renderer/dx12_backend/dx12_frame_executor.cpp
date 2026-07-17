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
        m_prevSubmissionSignals.clear();
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
        m_prevSubmissionSignals.clear();
        m_nextFenceValue = 1;
        m_nextQueueFenceValues = { 1, 1, 1 };
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
        // The GPU is idle: no previous-frame work remains to order against, and
        // the graph is typically rebuilt after a drain (submission indices would
        // no longer correspond). Drop the stale signals.
        m_prevSubmissionSignals.clear();
    }

    bool DX12FrameExecutor::submitAndPresent(
        const CompiledGraphPlan& plan,
        const std::vector<ID3D12CommandList*>& commandLists,
        uint32_t frameSlot,
        const GraphExecutionContext& execution,
        const DX12UploadDependency& uploadDependency)
    {
        const auto queueIndex = [](QueueType queue)
        {
            return static_cast<uint32_t>(queue);
        };

        // Translate the compiler's cross-frame edges into per-submission waits on
        // the PREVIOUS frame's producing submission. Only edges whose consumer
        // node actually executes this frame are applied (a cadence-skipped writer
        // creates no hazard), and only when the previous-frame signal map still
        // lines up with this plan (it is dropped on a GPU drain / graph rebuild).
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
        // Per-frame uploads (model geometry/textures) are consumed ONLY by
        // graphics scene-geometry draw passes, so only the graphics queue needs
        // to synchronize with the upload. With the current DX12UploadScheduler
        // the upload runs on the graphics queue itself, so this is pure
        // submission order (no fence wait); were the upload moved to a dedicated
        // queue, the graphics queue would wait on its fence here.
        //
        // Critically, the wait must NOT be applied to the compute/copy queues:
        // no async compute/copy pass reads scheduler-uploaded data (the cluster
        // passes read CPU-mapped upload-heap constants/lights, not model
        // geometry), and making the async compute queue wait on the graphics
        // upload fence synchronizes it after the upload's graphics-queue
        // resource-state transitions, which triggers DXGI_ERROR_ACCESS_DENIED
        // device removal. A non-graphics consumer, if ever added, must be modeled
        // as a frame-graph transfer pass so the normal cross-queue fence
        // machinery orders it explicitly.
        bool uploadWaitApplied = false;
        auto waitForUploads =
            [&](ID3D12CommandQueue* queue, QueueType queueType)
            {
                if (!uploadDependency.valid() || uploadWaitApplied ||
                    queueType != QueueType::Graphics)
                {
                    return;
                }
                uploadWaitApplied = true;
                if (queue != uploadDependency.queue)
                {
                    throwIfFailed(
                        queue->Wait(
                            uploadDependency.fence,
                            uploadDependency.value),
                        "Failed to enqueue DX12 upload dependency.");
                }
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
                spdlog::info(
                    "[DX12FrameExecutor] cross-frame deps={} "
                    "(0 => consecutive frames overlap fully; the old blanket "
                    "previous-frame barrier is removed)",
                    plan.crossFrameDependencies.size());
                loggedSchedule = true;
            }
            m_commandIndexByNode.assign(plan.nodes.size(), UINT32_MAX);
            for (uint32_t i = 0;
                 i < plan.executionLevelNodes.size(); ++i)
            {
                m_commandIndexByNode[plan.executionLevelNodes[i]] = i;
            }

            m_submissionSignals.assign(
                plan.queueSubmissions.size(), CrossFrameSignal{});
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

                // Minimal cross-frame ordering: wait only for the specific prior-
                // frame submissions that last used a resource this submission now
                // reuses. Cross-frame hazards have no intra-frame barrier between
                // them, so even a same-queue wait is meaningful here (unlike the
                // intra-frame same-queue waits below, which submission order
                // already satisfies).
                for (const CrossFrameSignal& wait :
                     m_crossFrameWaits[submissionIndex])
                {
                    throwIfFailed(
                        queue->Wait(
                            m_queueFences[queueIndex(wait.queue)].Get(),
                            wait.value),
                        "Failed to enqueue DX12 cross-frame wait.");
                }

                for (uint32_t i = 0; i < submission.waitCount; ++i)
                {
                    const uint32_t dependency =
                        plan.queueSubmissionWaits[
                            submission.firstWait + i].submissionIndex;
                    if (dependency < m_submissionSignals.size() &&
                        m_submissionSignals[dependency].value != 0)
                    {
                        const CrossFrameSignal& source =
                            m_submissionSignals[dependency];
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

                m_batchLists.clear();
                m_batchLists.reserve(submission.nodeCount);
                for (uint32_t i = 0; i < submission.nodeCount; ++i)
                {
                    const GraphNodeId node =
                        plan.queueSubmissionNodes[submission.firstNode + i];
                    const uint32_t commandIndex = m_commandIndexByNode[node];
                    if (commandIndex != UINT32_MAX &&
                        commandIndex < commandLists.size())
                    {
                        m_batchLists.push_back(commandLists[commandIndex]);
                    }
                }
                if (!m_batchLists.empty())
                {
                    queue->ExecuteCommandLists(
                        static_cast<UINT>(m_batchLists.size()),
                        m_batchLists.data());
                }

                const uint32_t signalQueue = queueIndex(submission.queue);
                const uint64_t signal =
                    m_nextQueueFenceValues[signalQueue]++;
                throwIfFailed(
                    queue->Signal(m_queueFences[signalQueue].Get(), signal),
                    "Failed to signal DX12 queue submission.");
                m_submissionSignals[submissionIndex] = {
                    submission.queue, signal };
                lastQueueSignals[signalQueue] = signal;

            }

            // Publish this frame's per-submission signals so the next frame's
            // cross-frame edges can wait on the exact producing submission.
            m_prevSubmissionSignals.assign(
                plan.queueSubmissions.size(), CrossFrameSignal{});
            for (uint32_t i = 0; i < m_submissionSignals.size(); ++i)
            {
                m_prevSubmissionSignals[i] = {
                    m_submissionSignals[i].queue, m_submissionSignals[i].value };
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
