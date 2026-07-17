// ic/renderer/dx12_backend/dx12_gpu_profiler.h
#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <span>
#include <vector>

#include <d3d12.h>
#include <wrl/client.h>

#include "ic/renderer/gpu_queue_profiler.h"
#include "dx12_device.h"
#include "dx12_resource_allocator.h"

namespace ic
{
    // Per-pass GPU timestamps for the DX12 backend, resolved into the shared
    // CPU-millisecond domain of GpuPassSample.
    //
    // Each pass records its own begin/end timestamp into its own command list
    // and resolves its own pair, which keeps the profiler compatible with the
    // parallel per-node recording the backend already does (slot allocation is
    // atomic; nothing else is shared between workers).
    //
    // Storage is per queue rather than shared. A single resolve target written
    // from both the graphics and compute queues in the same frame is flagged by
    // the debug layer as simultaneous cross-queue access even when the queues
    // write disjoint ranges, and the whole point of this profiler is to run
    // while async compute is active.
    //
    // Results lag by framesInFlight frames: a slot's readback is only safe to
    // read once that slot's fence has been waited on, which is exactly what
    // beginFrame() assumes its caller has already done.
    class DX12GpuProfiler
    {
    public:
        void init(
            const DX12Device& device,
            DX12ResourceAllocator& allocator,
            uint32_t framesInFlight,
            uint32_t maxPassesPerFrame);
        void shutdown();

        void setEnabled(bool enabled)
        {
            m_enabled = enabled && m_storage[0].queryHeap &&
                m_storage[1].queryHeap;
        }
        [[nodiscard]] bool enabled() const { return m_enabled; }

        // Resolves the timestamps recorded into this slot framesInFlight frames
        // ago into lastFrameSamples(), then re-arms the slot. The caller must
        // already have waited on this slot's frame fence.
        void beginFrame(uint32_t frameSlot);

        // Claims a query pair and writes the begin timestamp. Returns the slot
        // index needed by endPass, or UINT32_MAX when profiling is off or the
        // per-frame budget is exhausted (recording then proceeds untimed).
        uint32_t beginPass(
            ID3D12GraphicsCommandList4* cmd,
            uint32_t frameSlot,
            GraphNodeId node,
            QueueType queue);

        void endPass(
            ID3D12GraphicsCommandList4* cmd,
            uint32_t frameSlot,
            QueueType queue,
            uint32_t passSlot);

        [[nodiscard]] std::span<const GpuPassSample> lastFrameSamples() const
        {
            return m_lastFrameSamples;
        }

    private:
        // Only graphics and compute are timed; transfer passes are excluded
        // because a copy queue needs a different query heap type and never
        // participates in the async-compute decision.
        static constexpr uint32_t kTimedQueueCount = 2;

        struct PassRecord
        {
            GraphNodeId node{};
        };

        // GPU->CPU clock mapping for one queue. D3D12 exposes both the
        // frequency and the calibration per queue, and on several vendors the
        // compute queue's timestamp frequency differs from the graphics
        // queue's, so raw ticks from different queues are meaningless to
        // compare directly. Everything is mapped through here first.
        struct QueueClock
        {
            uint64_t frequency = 0;
            uint64_t gpuTimestamp = 0;
            double cpuMilliseconds = 0.0;
            bool valid = false;
        };

        struct QueueStorage
        {
            Microsoft::WRL::ComPtr<ID3D12QueryHeap> queryHeap;
            DX12Buffer readback{};
        };

        void calibrate();
        [[nodiscard]] double toCpuMilliseconds(
            QueueType queue, uint64_t gpuTicks) const;
        [[nodiscard]] static bool timedQueue(QueueType queue)
        {
            return queue == QueueType::Graphics || queue == QueueType::Compute;
        }
        [[nodiscard]] uint32_t queryBase(
            uint32_t frameSlot, uint32_t passSlot) const
        {
            return (frameSlot * m_maxPasses + passSlot) * 2u;
        }
        [[nodiscard]] uint32_t cursorIndex(
            uint32_t frameSlot, uint32_t queue) const
        {
            return frameSlot * kTimedQueueCount + queue;
        }
        [[nodiscard]] size_t recordIndex(
            uint32_t frameSlot, uint32_t queue, uint32_t passSlot) const
        {
            return (static_cast<size_t>(cursorIndex(frameSlot, queue)) *
                m_maxPasses) + passSlot;
        }

        const DX12Device* m_device = nullptr;
        DX12ResourceAllocator* m_allocator = nullptr;

        std::array<QueueStorage, kTimedQueueCount> m_storage{};

        uint32_t m_framesInFlight = 0;
        uint32_t m_maxPasses = 0;
        bool m_enabled = true;

        // One atomic cursor per (frame slot, queue): parallel recorders claim
        // pass slots without a lock, and the cursor doubles as that queue's
        // pass count when the results are read back a frame later.
        std::vector<std::atomic<uint32_t>> m_cursors;
        std::vector<PassRecord> m_records;

        std::array<QueueClock, kTimedQueueCount> m_queueClocks{};
        int64_t m_qpcOrigin = 0;
        double m_qpcFrequency = 0.0;
        uint32_t m_framesSinceCalibration = 0;

        std::vector<GpuPassSample> m_lastFrameSamples;
    };
}
