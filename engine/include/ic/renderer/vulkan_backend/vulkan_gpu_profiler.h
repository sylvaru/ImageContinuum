// ic/renderer/vulkan_backend/vulkan_gpu_profiler.h
#pragma once
#include <atomic>
#include <cstdint>
#include <span>
#include <vector>

#include "ic/renderer/gpu_queue_profiler.h"
#include "vulkan_platform.h"

namespace ic
{
    // Per-pass GPU timestamps for the Vulkan backend, mirroring
    // DX12GpuProfiler so both backends feed the same overlap analysis.
    //
    // Vulkan is simpler than D3D12 in two ways: a device exposes a single
    // timestamp domain and one timestampPeriod for every queue, so no
    // per-queue calibration is needed for the queues to be comparable; and
    // results are read straight to host memory, so no readback buffer or
    // resolve command is required.
    class VulkanGpuProfiler
    {
    public:
        void init(
            VkDevice device,
            double timestampPeriodNanoseconds,
            uint32_t timestampValidBits,
            uint32_t framesInFlight,
            uint32_t maxPassesPerFrame);
        void shutdown();

        void setEnabled(bool enabled)
        {
            m_enabled = enabled && m_queryPool != VK_NULL_HANDLE;
        }
        [[nodiscard]] bool enabled() const { return m_enabled; }

        // Reads back the timestamps this slot recorded framesInFlight frames
        // ago. The caller must already have waited on the slot's fence.
        void beginFrame(uint32_t frameSlot);

        uint32_t beginPass(
            VkCommandBuffer cmd,
            uint32_t frameSlot,
            GraphNodeId node,
            QueueType queue);

        void endPass(
            VkCommandBuffer cmd,
            uint32_t frameSlot,
            uint32_t passSlot);

        [[nodiscard]] std::span<const GpuPassSample> lastFrameSamples() const
        {
            return m_lastFrameSamples;
        }

    private:
        struct PassRecord
        {
            GraphNodeId node{};
            QueueType queue = QueueType::Graphics;
        };

        [[nodiscard]] uint32_t queryBase(
            uint32_t frameSlot, uint32_t passSlot) const
        {
            return (frameSlot * m_maxPasses + passSlot) * 2u;
        }

        [[nodiscard]] double toMillisecondsRelative(
            uint64_t ticks, uint64_t origin) const;

        VkDevice m_device = VK_NULL_HANDLE;
        VkQueryPool m_queryPool = VK_NULL_HANDLE;

        double m_timestampPeriodNanoseconds = 0.0;
        uint64_t m_timestampMask = ~uint64_t{ 0 };
        uint32_t m_framesInFlight = 0;
        uint32_t m_maxPasses = 0;
        bool m_enabled = true;

        std::vector<std::atomic<uint32_t>> m_cursors;
        std::vector<PassRecord> m_records;
        // Two uint64_t values (result + availability) per query, two queries
        // per pass. Sized once and reused by the batched slot readback.
        std::vector<uint64_t> m_queryScratch;

        std::vector<GpuPassSample> m_lastFrameSamples;
    };
}
