#include "ic/renderer/dx12_backend/dx12_gpu_profiler.h"

#include <algorithm>
#include <spdlog/spdlog.h>

namespace ic
{
    namespace
    {
        uint32_t queueIndex(QueueType queue)
        {
            return static_cast<uint32_t>(queue);
        }
    }

    void DX12GpuProfiler::init(
        const DX12Device& device,
        DX12ResourceAllocator& allocator,
        uint32_t framesInFlight,
        uint32_t maxPassesPerFrame)
    {
        m_device = &device;
        m_allocator = &allocator;
        m_framesInFlight = std::max(1u, framesInFlight);
        m_maxPasses = std::max(1u, maxPassesPerFrame);

        const uint32_t queryCount = m_framesInFlight * m_maxPasses * 2u;

        for (uint32_t queue = 0; queue < kTimedQueueCount; ++queue)
        {
            D3D12_QUERY_HEAP_DESC desc{};
            desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
            desc.Count = queryCount;
            if (FAILED(device.device()->CreateQueryHeap(
                    &desc, IID_PPV_ARGS(&m_storage[queue].queryHeap))))
            {
                spdlog::warn(
                    "[DX12GpuProfiler] Timestamp heap creation failed; "
                    "GPU pass timing disabled.");
                m_enabled = false;
                return;
            }

            m_storage[queue].readback = allocator.createBuffer({
                .size = sizeof(uint64_t) * queryCount,
                .usage = BufferUsageFlags::TransferDst,
                .memoryUsage = ResourceMemoryUsage::GpuToCpu,
                .mappedAtCreation = true,
                .debugName = queue == 0
                    ? "GPU pass timestamp readback (graphics)"
                    : "GPU pass timestamp readback (compute)"
            });
        }

        // std::atomic is not movable, so the cursor vector is sized in place.
        m_cursors = std::vector<std::atomic<uint32_t>>(
            static_cast<size_t>(m_framesInFlight) * kTimedQueueCount);
        for (auto& cursor : m_cursors)
        {
            cursor.store(0, std::memory_order_relaxed);
        }
        m_records.assign(
            static_cast<size_t>(m_framesInFlight) * kTimedQueueCount *
                m_maxPasses,
            PassRecord{});
        m_lastFrameSamples.reserve(
            static_cast<size_t>(m_maxPasses) * kTimedQueueCount);

        LARGE_INTEGER qpcFrequency{};
        QueryPerformanceFrequency(&qpcFrequency);
        m_qpcFrequency = static_cast<double>(qpcFrequency.QuadPart);
        LARGE_INTEGER qpcNow{};
        QueryPerformanceCounter(&qpcNow);
        m_qpcOrigin = qpcNow.QuadPart;

        calibrate();
        m_framesSinceCalibration = 0;
    }

    void DX12GpuProfiler::shutdown()
    {
        for (QueueStorage& storage : m_storage)
        {
            if (m_allocator && storage.readback.resource)
            {
                m_allocator->destroyBuffer(storage.readback);
            }
            storage.readback = {};
            storage.queryHeap.Reset();
        }
        m_cursors.clear();
        m_records.clear();
        m_lastFrameSamples.clear();
        m_device = nullptr;
        m_allocator = nullptr;
    }

    void DX12GpuProfiler::calibrate()
    {
        if (!m_device)
        {
            return;
        }

        const auto calibrateQueue =
            [&](QueueType queue, ID3D12CommandQueue* commandQueue)
            {
                QueueClock& clock = m_queueClocks[queueIndex(queue)];
                clock.valid = false;
                if (!commandQueue)
                {
                    return;
                }

                uint64_t gpuTimestamp = 0;
                uint64_t cpuTimestamp = 0;
                if (FAILED(commandQueue->GetTimestampFrequency(
                        &clock.frequency)) ||
                    clock.frequency == 0 ||
                    FAILED(commandQueue->GetClockCalibration(
                        &gpuTimestamp, &cpuTimestamp)))
                {
                    return;
                }

                clock.gpuTimestamp = gpuTimestamp;
                clock.cpuMilliseconds =
                    static_cast<double>(
                        static_cast<int64_t>(cpuTimestamp) - m_qpcOrigin) *
                    1000.0 / m_qpcFrequency;
                clock.valid = true;
            };

        calibrateQueue(QueueType::Graphics, m_device->graphicsQueue());
        calibrateQueue(QueueType::Compute, m_device->computeQueue());
    }

    double DX12GpuProfiler::toCpuMilliseconds(
        QueueType queue, uint64_t gpuTicks) const
    {
        const QueueClock& clock = m_queueClocks[queueIndex(queue)];
        if (!clock.valid)
        {
            return 0.0;
        }

        const double deltaTicks =
            static_cast<double>(
                static_cast<int64_t>(gpuTicks) -
                static_cast<int64_t>(clock.gpuTimestamp));
        return clock.cpuMilliseconds +
            deltaTicks * 1000.0 / static_cast<double>(clock.frequency);
    }

    void DX12GpuProfiler::beginFrame(uint32_t frameSlot)
    {
        m_lastFrameSamples.clear();
        if (frameSlot >= m_framesInFlight || !m_storage[0].queryHeap)
        {
            return;
        }
        if (!m_enabled)
        {
            for (uint32_t queue = 0; queue < kTimedQueueCount; ++queue)
            {
                m_cursors[cursorIndex(frameSlot, queue)].store(
                    0, std::memory_order_relaxed);
            }
            return;
        }

        // Clock calibration crosses the user/kernel/driver boundary for each
        // queue. Refresh often enough to bound drift without paying those calls
        // every frame; pass durations use queue-local tick deltas either way.
        constexpr uint32_t kCalibrationIntervalFrames = 120u;
        if (++m_framesSinceCalibration >= kCalibrationIntervalFrames)
        {
            calibrate();
            m_framesSinceCalibration = 0;
        }

        for (uint32_t queue = 0; queue < kTimedQueueCount; ++queue)
        {
            // The cursor still holds the pass count from this slot's previous
            // use, which is exactly the number of query pairs written back.
            // Reading it here (before the reset below) avoids a second, racy
            // counter.
            const uint32_t passCount = std::min(
                m_cursors[cursorIndex(frameSlot, queue)].load(
                    std::memory_order_relaxed),
                m_maxPasses);
            if (passCount == 0 || !m_storage[queue].readback.mapped)
            {
                continue;
            }

            auto* timestamps =
                static_cast<uint64_t*>(m_storage[queue].readback.mapped);
            const QueueType queueType = static_cast<QueueType>(queue);

            for (uint32_t pass = 0; pass < passCount; ++pass)
            {
                const PassRecord& record =
                    m_records[recordIndex(frameSlot, queue, pass)];
                const uint32_t base = queryBase(frameSlot, pass);
                const uint64_t begin = timestamps[base];
                const uint64_t end = timestamps[base + 1u];

                // A pass that claimed a slot but never resolved (skipped or
                // budget-truncated) leaves the pair untouched; the clear below
                // makes that read back as zeroes rather than as a stale
                // interval from an older frame.
                if (end > begin)
                {
                    m_lastFrameSamples.push_back({
                        .node = record.node,
                        .queue = queueType,
                        .beginMs = toCpuMilliseconds(queueType, begin),
                        .endMs = toCpuMilliseconds(queueType, end)
                    });
                }
            }

            std::fill_n(
                timestamps + queryBase(frameSlot, 0),
                static_cast<size_t>(passCount) * 2u,
                uint64_t{ 0 });
        }

        for (uint32_t queue = 0; queue < kTimedQueueCount; ++queue)
        {
            m_cursors[cursorIndex(frameSlot, queue)].store(
                0, std::memory_order_relaxed);
        }
    }

    uint32_t DX12GpuProfiler::beginPass(
        ID3D12GraphicsCommandList4* cmd,
        uint32_t frameSlot,
        GraphNodeId node,
        QueueType queue)
    {
        if (!m_enabled || !cmd || frameSlot >= m_framesInFlight ||
            !timedQueue(queue) || !m_storage[queueIndex(queue)].queryHeap)
        {
            return UINT32_MAX;
        }

        const uint32_t queueSlot = queueIndex(queue);
        const uint32_t passSlot =
            m_cursors[cursorIndex(frameSlot, queueSlot)].fetch_add(
                1, std::memory_order_relaxed);
        if (passSlot >= m_maxPasses)
        {
            return UINT32_MAX;
        }

        m_records[recordIndex(frameSlot, queueSlot, passSlot)] = { node };

        cmd->EndQuery(
            m_storage[queueSlot].queryHeap.Get(),
            D3D12_QUERY_TYPE_TIMESTAMP,
            queryBase(frameSlot, passSlot));
        return passSlot;
    }

    void DX12GpuProfiler::endPass(
        ID3D12GraphicsCommandList4* cmd,
        uint32_t frameSlot,
        QueueType queue,
        uint32_t passSlot)
    {
        if (passSlot == UINT32_MAX || !cmd || frameSlot >= m_framesInFlight ||
            !timedQueue(queue) || !m_storage[queueIndex(queue)].queryHeap)
        {
            return;
        }

        const uint32_t queueSlot = queueIndex(queue);
        const uint32_t base = queryBase(frameSlot, passSlot);
        cmd->EndQuery(
            m_storage[queueSlot].queryHeap.Get(),
            D3D12_QUERY_TYPE_TIMESTAMP,
            base + 1u);

        // Each pass resolves its own pair so no extra command list or
        // submission is introduced purely for profiling.
        cmd->ResolveQueryData(
            m_storage[queueSlot].queryHeap.Get(),
            D3D12_QUERY_TYPE_TIMESTAMP,
            base,
            2u,
            m_storage[queueSlot].readback.resource.Get(),
            sizeof(uint64_t) * base);
    }
}
