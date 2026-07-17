#include "ic/renderer/vulkan_backend/vulkan_gpu_profiler.h"

#include <algorithm>
#include <array>
#include <spdlog/spdlog.h>

namespace ic
{
    void VulkanGpuProfiler::init(
        VkDevice device,
        double timestampPeriodNanoseconds,
        uint32_t timestampValidBits,
        uint32_t framesInFlight,
        uint32_t maxPassesPerFrame)
    {
        m_device = device;
        m_timestampPeriodNanoseconds = timestampPeriodNanoseconds;
        m_framesInFlight = std::max(1u, framesInFlight);
        m_maxPasses = std::max(1u, maxPassesPerFrame);

        // A queue family may implement fewer than 64 valid timestamp bits; the
        // undefined high bits must be masked off before any arithmetic.
        m_timestampMask = (timestampValidBits == 0u || timestampValidBits >= 64u)
            ? ~uint64_t{ 0 }
            : ((uint64_t{ 1 } << timestampValidBits) - uint64_t{ 1 });

        if (timestampValidBits == 0u || timestampPeriodNanoseconds <= 0.0)
        {
            spdlog::warn(
                "[VulkanGpuProfiler] Queue reports no valid timestamp bits; "
                "GPU pass timing disabled.");
            m_enabled = false;
            return;
        }

        VkQueryPoolCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        info.queryCount = m_framesInFlight * m_maxPasses * 2u;
        if (vkCreateQueryPool(m_device, &info, nullptr, &m_queryPool) !=
            VK_SUCCESS)
        {
            spdlog::warn(
                "[VulkanGpuProfiler] Timestamp query pool creation failed; "
                "GPU pass timing disabled.");
            m_queryPool = VK_NULL_HANDLE;
            m_enabled = false;
            return;
        }

        m_cursors = std::vector<std::atomic<uint32_t>>(m_framesInFlight);
        for (auto& cursor : m_cursors)
        {
            cursor.store(0, std::memory_order_relaxed);
        }
        m_records.assign(
            static_cast<size_t>(m_framesInFlight) * m_maxPasses, PassRecord{});
        m_queryScratch.resize(static_cast<size_t>(m_maxPasses) * 4u);
        m_lastFrameSamples.reserve(m_maxPasses);
    }

    void VulkanGpuProfiler::shutdown()
    {
        if (m_queryPool != VK_NULL_HANDLE && m_device != VK_NULL_HANDLE)
        {
            vkDestroyQueryPool(m_device, m_queryPool, nullptr);
        }
        m_queryPool = VK_NULL_HANDLE;
        m_device = VK_NULL_HANDLE;
        m_cursors.clear();
        m_records.clear();
        m_queryScratch.clear();
        m_lastFrameSamples.clear();
    }

    double VulkanGpuProfiler::toMillisecondsRelative(
        uint64_t ticks, uint64_t origin) const
    {
        const uint64_t masked = ticks & m_timestampMask;
        const uint64_t maskedOrigin = origin & m_timestampMask;
        double delta = 0.0;
        if (m_timestampMask == ~uint64_t{ 0 })
        {
            delta = static_cast<double>(
                static_cast<int64_t>(masked - maskedOrigin));
        }
        else
        {
            const uint64_t forward =
                (masked - maskedOrigin) & m_timestampMask;
            if (forward <= m_timestampMask / 2u)
            {
                delta = static_cast<double>(forward);
            }
            else
            {
                const uint64_t backward =
                    (maskedOrigin - masked) & m_timestampMask;
                delta = -static_cast<double>(backward);
            }
        }
        return delta * m_timestampPeriodNanoseconds / 1.0e6;
    }

    void VulkanGpuProfiler::beginFrame(uint32_t frameSlot)
    {
        m_lastFrameSamples.clear();
        if (m_queryPool == VK_NULL_HANDLE || frameSlot >= m_framesInFlight)
        {
            return;
        }
        if (!m_enabled)
        {
            m_cursors[frameSlot].store(0, std::memory_order_relaxed);
            return;
        }

        const uint32_t passCount = std::min(
            m_cursors[frameSlot].load(std::memory_order_relaxed), m_maxPasses);

        if (passCount > 0)
        {
            const VkResult result = vkGetQueryPoolResults(
                m_device,
                m_queryPool,
                queryBase(frameSlot, 0u),
                passCount * 2u,
                static_cast<size_t>(passCount) * 4u * sizeof(uint64_t),
                m_queryScratch.data(),
                sizeof(uint64_t) * 2u,
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
            if (result == VK_SUCCESS)
            {
                uint64_t origin = 0u;
                bool haveOrigin = false;
                for (uint32_t pass = 0; pass < passCount; ++pass)
                {
                    const size_t base = static_cast<size_t>(pass) * 4u;
                    if (m_queryScratch[base + 1u] != 0u)
                    {
                        origin = m_queryScratch[base];
                        haveOrigin = true;
                        break;
                    }
                }
                if (!haveOrigin)
                {
                    m_cursors[frameSlot].store(0, std::memory_order_relaxed);
                    return;
                }
                for (uint32_t pass = 0; pass < passCount; ++pass)
                {
                    const size_t resultBase = static_cast<size_t>(pass) * 4u;
                    if (m_queryScratch[resultBase + 1u] == 0u ||
                        m_queryScratch[resultBase + 3u] == 0u)
                    {
                        continue;
                    }

                    const double beginMs =
                        toMillisecondsRelative(
                            m_queryScratch[resultBase], origin);
                    const double endMs =
                        toMillisecondsRelative(
                            m_queryScratch[resultBase + 2u], origin);
                    if (!(endMs > beginMs))
                    {
                        continue;
                    }

                    const PassRecord& record = m_records[
                        static_cast<size_t>(frameSlot) * m_maxPasses + pass];
                    m_lastFrameSamples.push_back({
                        .node = record.node,
                        .queue = record.queue,
                        .beginMs = beginMs,
                        .endMs = endMs
                    });
                }
            }
        }

        m_cursors[frameSlot].store(0, std::memory_order_relaxed);
    }

    uint32_t VulkanGpuProfiler::beginPass(
        VkCommandBuffer cmd,
        uint32_t frameSlot,
        GraphNodeId node,
        QueueType queue)
    {
        if (!m_enabled || m_queryPool == VK_NULL_HANDLE ||
            cmd == VK_NULL_HANDLE || frameSlot >= m_framesInFlight ||
            queue == QueueType::Transfer)
        {
            return UINT32_MAX;
        }

        const uint32_t passSlot =
            m_cursors[frameSlot].fetch_add(1, std::memory_order_relaxed);
        if (passSlot >= m_maxPasses)
        {
            return UINT32_MAX;
        }

        m_records[static_cast<size_t>(frameSlot) * m_maxPasses + passSlot] = {
            .node = node, .queue = queue };

        const uint32_t base = queryBase(frameSlot, passSlot);
        // Each pass owns a disjoint pair, so resetting here (outside any render
        // pass, at the top of this pass's own command buffer) cannot disturb
        // another worker's queries.
        vkCmdResetQueryPool(cmd, m_queryPool, base, 2u);
        vkCmdWriteTimestamp(
            cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_queryPool, base);
        return passSlot;
    }

    void VulkanGpuProfiler::endPass(
        VkCommandBuffer cmd,
        uint32_t frameSlot,
        uint32_t passSlot)
    {
        if (passSlot == UINT32_MAX || m_queryPool == VK_NULL_HANDLE ||
            cmd == VK_NULL_HANDLE || frameSlot >= m_framesInFlight)
        {
            return;
        }

        vkCmdWriteTimestamp(
            cmd,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            m_queryPool,
            queryBase(frameSlot, passSlot) + 1u);
    }
}
