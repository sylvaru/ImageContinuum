#include "ic/common/ic_pch.h"
#include "ic/renderer/ibl_baker.h"

#include <algorithm>

#include <spdlog/spdlog.h>

namespace ic
{
    IBLHandle IBLBaker::requestBake(const IBLBakeDesc& desc)
    {
        if (!desc.sourceEnvironment)
        {
            return {};
        }

        std::scoped_lock lock(m_mutex);

        for (uint32_t i = 0; i < m_records.size(); ++i)
        {
            const ProbeRecord& candidate = m_records[i];
            if (candidate.state != IBLBakeState::Empty &&
                sameBakeKey(candidate.desc, desc))
            {
                return IBLHandle{ i, candidate.generation };
            }
        }

        uint32_t index = 0;
        if (!m_freeList.empty())
        {
            index = m_freeList.back();
            m_freeList.pop_back();
        }
        else
        {
            index = static_cast<uint32_t>(m_records.size());
            m_records.emplace_back();
        }

        return enqueueLocked(index, desc);
    }

    IBLHandle IBLBaker::requestRebake(
        IBLHandle previous,
        const IBLBakeDesc& desc)
    {
        if (!desc.sourceEnvironment)
        {
            return {};
        }

        std::scoped_lock lock(m_mutex);

        ProbeRecord* existing = record(previous);
        if (!existing)
        {
            uint32_t index = 0;
            if (!m_freeList.empty())
            {
                index = m_freeList.back();
                m_freeList.pop_back();
            }
            else
            {
                index = static_cast<uint32_t>(m_records.size());
                m_records.emplace_back();
            }

            return enqueueLocked(index, desc);
        }

        ++existing->generation;
        existing->desc = desc;
        existing->resources = {};
        existing->state = IBLBakeState::Queued;
        existing->error.clear();
        existing->requestId = m_nextRequestId++;

        IBLHandle next{ previous.index, existing->generation };
        m_pendingRequests.push_back(
            IBLBakeRequest{
                .requestId = existing->requestId,
                .handle = next,
                .desc = desc });

        spdlog::info(
            "[IBLBaker] Queued rebake '{}' request={} handle={}:{} "
            "source={}:{} env={} irradiance={} prefilter={} brdfLut={}",
            desc.debugName,
            existing->requestId,
            next.index,
            next.generation,
            desc.sourceEnvironment.index,
            desc.sourceEnvironment.generation,
            desc.environmentSize,
            desc.irradianceSize,
            desc.prefilterSize,
            desc.brdfLutSize);
        return next;
    }

    IBLBakeState IBLBaker::state(IBLHandle handle) const
    {
        std::scoped_lock lock(m_mutex);
        const ProbeRecord* probe = record(handle);
        return probe ? probe->state : IBLBakeState::Empty;
    }

    bool IBLBaker::resources(
        IBLHandle handle,
        IBLResources& outResources) const
    {
        std::scoped_lock lock(m_mutex);
        const ProbeRecord* probe = record(handle);
        if (!probe || probe->state != IBLBakeState::Ready)
        {
            return false;
        }

        outResources = probe->resources;
        return true;
    }

    IBLProbeSnapshot IBLBaker::snapshot(IBLHandle handle) const
    {
        std::scoped_lock lock(m_mutex);
        const ProbeRecord* probe = record(handle);
        if (!probe)
        {
            return {};
        }

        return IBLProbeSnapshot{
            .handle = handle,
            .desc = probe->desc,
            .resources = probe->resources,
            .state = probe->state,
            .error = probe->error,
            .requestId = probe->requestId };
    }

    std::vector<IBLBakeRequest> IBLBaker::takePendingRequests()
    {
        std::scoped_lock lock(m_mutex);
        std::vector<IBLBakeRequest> requests;
        requests.swap(m_pendingRequests);

        for (const IBLBakeRequest& request : requests)
        {
            ProbeRecord* probe = record(request.handle);
            if (probe && probe->requestId == request.requestId)
            {
                probe->state = IBLBakeState::WaitingForSource;
                spdlog::info(
                    "[IBLBaker] Dispatching bake '{}' request={} handle={}:{}",
                    request.desc.debugName,
                    request.requestId,
                    request.handle.index,
                    request.handle.generation);
            }
        }

        return requests;
    }

    bool IBLBaker::requeue(IBLHandle handle)
    {
        std::scoped_lock lock(m_mutex);
        ProbeRecord* probe = record(handle);
        if (!probe ||
            (probe->state != IBLBakeState::WaitingForSource &&
                probe->state != IBLBakeState::Queued))
        {
            return false;
        }

        const bool alreadyQueued =
            std::ranges::any_of(
                m_pendingRequests,
                [handle](const IBLBakeRequest& request)
                {
                    return request.handle == handle;
                });
        if (alreadyQueued)
        {
            return false;
        }

        probe->state = IBLBakeState::Queued;
        m_pendingRequests.push_back(
            IBLBakeRequest{
                .requestId = probe->requestId,
                .handle = handle,
                .desc = probe->desc });
        return true;
    }

    void IBLBaker::markUploading(
        IBLHandle handle,
        uint64_t requestId)
    {
        std::scoped_lock lock(m_mutex);
        ProbeRecord* probe = record(handle);
        if (probe && probe->requestId == requestId)
        {
            probe->state = IBLBakeState::UploadingGPU;
            spdlog::info(
                "[IBLBaker] Uploading source environment for request={} "
                "handle={}:{}",
                requestId,
                handle.index,
                handle.generation);
        }
    }

    void IBLBaker::markBaking(
        IBLHandle handle,
        uint64_t requestId)
    {
        std::scoped_lock lock(m_mutex);
        ProbeRecord* probe = record(handle);
        if (probe && probe->requestId == requestId)
        {
            probe->state = IBLBakeState::Baking;
            spdlog::info(
                "[IBLBaker] Baking GPU IBL resources for request={} "
                "handle={}:{}",
                requestId,
                handle.index,
                handle.generation);
        }
    }

    void IBLBaker::markWaitingForSource(
        IBLHandle handle,
        uint64_t requestId)
    {
        std::scoped_lock lock(m_mutex);
        ProbeRecord* probe = record(handle);
        if (probe && probe->requestId == requestId)
        {
            probe->state = IBLBakeState::WaitingForSource;
        }
    }

    void IBLBaker::publishResult(const IBLBakeResult& result)
    {
        std::scoped_lock lock(m_mutex);
        ProbeRecord* probe = record(result.handle);
        if (!probe || probe->requestId != result.requestId)
        {
            return;
        }

        if (result.success)
        {
            probe->resources = result.resources;
            probe->state = IBLBakeState::Ready;
            probe->error.clear();
            spdlog::info(
                "[IBLBaker] Bake completed '{}' request={} handle={}:{} "
                "environmentTex={} irradianceTex={} prefilteredTex={} "
                "brdfLutTex={} bindless=[{},{},{},{}] prefilteredMips={}",
                probe->desc.debugName,
                result.requestId,
                result.handle.index,
                result.handle.generation,
                result.resources.environmentCubemap.value,
                result.resources.irradianceCubemap.value,
                result.resources.prefilteredCubemap.value,
                result.resources.brdfLut.value,
                result.resources.environmentBindlessIndex,
                result.resources.irradianceBindlessIndex,
                result.resources.prefilteredBindlessIndex,
                result.resources.brdfLutBindlessIndex,
                result.resources.prefilteredMipCount);
        }
        else
        {
            probe->resources = {};
            probe->state = IBLBakeState::Failed;
            probe->error = result.error;
            spdlog::error(
                "[IBLBaker] Bake failed '{}' request={} handle={}:{}: {}",
                probe->desc.debugName,
                result.requestId,
                result.handle.index,
                result.handle.generation,
                probe->error);
        }
    }

    IBLBaker::ProbeRecord* IBLBaker::record(IBLHandle handle)
    {
        if (!handle || handle.index >= m_records.size())
        {
            return nullptr;
        }

        ProbeRecord& probe = m_records[handle.index];
        if (probe.generation != handle.generation ||
            probe.state == IBLBakeState::Empty)
        {
            return nullptr;
        }

        return &probe;
    }

    const IBLBaker::ProbeRecord* IBLBaker::record(IBLHandle handle) const
    {
        if (!handle || handle.index >= m_records.size())
        {
            return nullptr;
        }

        const ProbeRecord& probe = m_records[handle.index];
        if (probe.generation != handle.generation ||
            probe.state == IBLBakeState::Empty)
        {
            return nullptr;
        }

        return &probe;
    }

    bool IBLBaker::sameBakeKey(
        const IBLBakeDesc& lhs,
        const IBLBakeDesc& rhs)
    {
        return lhs.sourceEnvironment == rhs.sourceEnvironment &&
               lhs.environmentSize == rhs.environmentSize &&
               lhs.irradianceSize == rhs.irradianceSize &&
               lhs.prefilterSize == rhs.prefilterSize &&
               lhs.brdfLutSize == rhs.brdfLutSize &&
               lhs.format == rhs.format;
    }

    IBLHandle IBLBaker::enqueueLocked(
        uint32_t index,
        const IBLBakeDesc& desc)
    {
        ProbeRecord& probe = m_records[index];
        probe.generation = std::max(1u, probe.generation);
        probe.desc = desc;
        probe.resources = {};
        probe.state = IBLBakeState::Queued;
        probe.error.clear();
        probe.requestId = m_nextRequestId++;

        IBLHandle handle{ index, probe.generation };
        m_pendingRequests.push_back(
            IBLBakeRequest{
                .requestId = probe.requestId,
                .handle = handle,
                .desc = desc });

        spdlog::info(
            "[IBLBaker] Queued bake '{}' request={} handle={}:{} "
            "source={}:{} env={} irradiance={} prefilter={} brdfLut={}",
            desc.debugName,
            probe.requestId,
            handle.index,
            handle.generation,
            desc.sourceEnvironment.index,
            desc.sourceEnvironment.generation,
            desc.environmentSize,
            desc.irradianceSize,
            desc.prefilterSize,
            desc.brdfLutSize);

        return handle;
    }
}
