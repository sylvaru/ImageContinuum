#include "ic/renderer/dx12_backend/dx12_graph_resource_registry.h"

#include "ic/renderer/dx12_backend/dx12_device.h"
#include "ic/renderer/render_types.h"

#include <algorithm>
#include <cassert>
#include <utility>

#include <spdlog/spdlog.h>

namespace ic
{
    void DX12GraphResourceRegistry::init(
        const DX12Device& device,
        DX12ResourceAllocator& resourceAllocator,
        DX12DescriptorSystem& descriptorSystem,
        uint32_t framesInFlight)
    {
        m_device = device.device();
        m_resourceAllocator = &resourceAllocator;
        m_descriptorSystem = &descriptorSystem;
        m_framesInFlight = std::max(1u, framesInFlight);
        m_frameCounter = 0;
        m_retiredByFrameSlot.clear();
        m_retiredByFrameSlot.resize(m_framesInFlight);
        m_materializationGeneration = 0;
    }

    uint32_t DX12GraphResourceRegistry::instanceCountFor(
        ResourceMultiplicity multiplicity) const noexcept
    {
        return multiplicity == ResourceMultiplicity::Single
            ? 1u
            : m_framesInFlight;
    }

    uint32_t DX12GraphResourceRegistry::currentInstanceIndex(
        const DX12GraphResourceInstances& instances) const noexcept
    {
        const uint32_t count = static_cast<uint32_t>(instances.slots.size());
        // Per-frame-slot resources index by the executor's frame slot so they
        // align with every other frame-slot-indexed backend structure (command
        // lists, cached descriptor sets, retirement buckets). Only history
        // resources index by the monotonic submitted-frame counter. That is
        // what makes their "previous" skip-safe.
        if (instances.multiplicity == ResourceMultiplicity::History)
        {
            return currentFrameInstanceIndex(m_frameCounter, count);
        }
        return currentFrameInstanceIndex(m_frameSlot, count);
    }

    uint32_t DX12GraphResourceRegistry::previousInstanceIndex(
        const DX12GraphResourceInstances& instances) const noexcept
    {
        return previousFrameInstanceIndex(
            m_frameCounter, static_cast<uint32_t>(instances.slots.size()));
    }

    void DX12GraphResourceRegistry::shutdown()
    {
        for (auto& [id, instances] : m_entries)
        {
            for (DX12GraphResourceEntry& entry : instances.slots)
            {
                destroyEntry(entry);
            }
        }
        m_entries.clear();

        for (std::vector<DX12GraphResourceEntry>& retired : m_retiredByFrameSlot)
        {
            for (DX12GraphResourceEntry& entry : retired)
            {
                destroyEntry(entry);
            }
            retired.clear();
        }
        m_retiredByFrameSlot.clear();

        m_device = nullptr;
        m_resourceAllocator = nullptr;
        m_descriptorSystem = nullptr;
    }

    void DX12GraphResourceRegistry::reset() noexcept
    {
        for (auto& [id, instances] : m_entries)
        {
            for (DX12GraphResourceEntry& entry : instances.slots)
            {
                destroyEntry(entry);
            }
        }
        m_entries.clear();

        for (std::vector<DX12GraphResourceEntry>& retired : m_retiredByFrameSlot)
        {
            for (DX12GraphResourceEntry& entry : retired)
            {
                destroyEntry(entry);
            }
            retired.clear();
        }
    }

    void DX12GraphResourceRegistry::recycleFrameSlot(uint32_t frameSlot)
    {
        if (frameSlot >= m_retiredByFrameSlot.size())
        {
            return;
        }

        std::vector<DX12GraphResourceEntry>& retired =
            m_retiredByFrameSlot[frameSlot];
        for (DX12GraphResourceEntry& entry : retired)
        {
            destroyEntry(entry);
        }
        retired.clear();
    }

    void DX12GraphResourceRegistry::materialize(
        const CompiledGraphPlan& plan,
        uint32_t frameSlot,
        uint32_t defaultWidth,
        uint32_t defaultHeight,
        const DX12GraphResourceImports& imports)
    {
        validateFrameSlot(frameSlot);
        // Executor frame slot (drives per-frame-slot instance selection).
        m_frameSlot = frameSlot;
        // Monotonic submitted-frame count (drives history current/previous
        // selection), advanced once per recorded frame regardless of skips.
        ++m_frameCounter;
        const uint64_t generation = nextGeneration();

        for (const GraphResource& resource : plan.resources)
        {
            DX12GraphResourceInstances& instances = m_entries[resource.id];
            instances.materializationGeneration = generation;
            instances.multiplicity = resource.multiplicity;

            // Size the instance ring for this resource's multiplicity. Extra
            // slots (from a shrinking ring) are retired into the current frame
            // slot so they are freed only after this slot's GPU work completes.
            const uint32_t desiredCount =
                resource.ownership == ResourceOwnership::Imported
                    ? 1u
                    : instanceCountFor(resource.multiplicity);
            while (instances.slots.size() > desiredCount)
            {
                retireEntry(std::move(instances.slots.back()), frameSlot);
                instances.slots.pop_back();
            }
            if (instances.slots.size() < desiredCount)
            {
                instances.slots.resize(desiredCount);
            }

            // Only the current frame slot's instance is (re)materialized here;
            // the other slots are materialized when their own frame comes round.
            const uint32_t index = currentInstanceIndex(instances);
            DX12GraphResourceEntry& entry = instances.slots[index];

            const TextureDesc resolvedTexture =
                resource.type == GraphResourceType::Texture
                    ? resolvedTextureDesc(
                          resource.textureDesc, defaultWidth, defaultHeight)
                    : TextureDesc{};

            if (resource.ownership == ResourceOwnership::Imported)
            {
                updateImported(entry, resource, resolvedTexture, imports);
                continue;
            }

            if (entryMatches(entry, resource, resolvedTexture))
            {
                // Unchanged: keep native resource, descriptors and tracked
                // state exactly as they were the previous time this slot ran.
                entry.id = resource.id;
                entry.type = resource.type;
                entry.ownership = resource.ownership;
                entry.imported = resource.imported;
                continue;
            }

            // Recreation required. Retire the old native contents into the
            // frame-slot bucket so they are freed only after the executor has
            // waited this slot, instead of stalling the GPU here.
            retireEntry(
                std::exchange(entry, DX12GraphResourceEntry{}), frameSlot);

            entry.id = resource.id;
            entry.type = resource.type;
            entry.ownership = resource.ownership;
            entry.imported = resource.imported;

            materializeTransient(entry, resource, resolvedTexture);
        }

        for (auto it = m_entries.begin(); it != m_entries.end();)
        {
            if (it->second.materializationGeneration == generation)
            {
                ++it;
                continue;
            }

            for (DX12GraphResourceEntry& entry : it->second.slots)
            {
                retireEntry(std::move(entry), frameSlot);
            }
            it = m_entries.erase(it);
        }
    }

    DX12GraphResourceEntry* DX12GraphResourceRegistry::entry(
        GraphResourceId id) noexcept
    {
        auto it = m_entries.find(id);
        if (it == m_entries.end() || it->second.slots.empty())
        {
            return nullptr;
        }
        return &it->second.slots[currentInstanceIndex(it->second)];
    }

    const DX12GraphResourceEntry* DX12GraphResourceRegistry::entry(
        GraphResourceId id) const noexcept
    {
        auto it = m_entries.find(id);
        if (it == m_entries.end() || it->second.slots.empty())
        {
            return nullptr;
        }
        return &it->second.slots[currentInstanceIndex(it->second)];
    }

    DX12GraphResourceEntry* DX12GraphResourceRegistry::previousEntry(
        GraphResourceId id) noexcept
    {
        auto it = m_entries.find(id);
        if (it == m_entries.end() || it->second.slots.empty())
        {
            return nullptr;
        }
        return &it->second.slots[previousInstanceIndex(it->second)];
    }

    const DX12GraphResourceEntry* DX12GraphResourceRegistry::previousEntry(
        GraphResourceId id) const noexcept
    {
        auto it = m_entries.find(id);
        if (it == m_entries.end() || it->second.slots.empty())
        {
            return nullptr;
        }
        return &it->second.slots[previousInstanceIndex(it->second)];
    }

    ID3D12Resource* DX12GraphResourceRegistry::nativeResource(
        GraphResourceId id) const noexcept
    {
        const DX12GraphResourceEntry* found = entry(id);
        return found ? found->nativeResource() : nullptr;
    }

    D3D12_CPU_DESCRIPTOR_HANDLE DX12GraphResourceRegistry::rtvHandle(
        GraphResourceId id) const noexcept
    {
        const DX12GraphResourceEntry* found = entry(id);
        return found ? found->rtvHandle() : D3D12_CPU_DESCRIPTOR_HANDLE{};
    }

    TextureDesc DX12GraphResourceRegistry::resolvedTextureDesc(
        const TextureDesc& desc,
        uint32_t defaultWidth,
        uint32_t defaultHeight) noexcept
    {
        TextureDesc resolved = desc;
        if (resolved.width == 0) resolved.width = defaultWidth;
        if (resolved.height == 0) resolved.height = defaultHeight;
        return resolved;
    }

    bool DX12GraphResourceRegistry::textureDescMatches(
        const TextureDesc& lhs,
        const TextureDesc& rhs) noexcept
    {
        return lhs.width == rhs.width &&
               lhs.height == rhs.height &&
               lhs.depth == rhs.depth &&
               lhs.mipLevels == rhs.mipLevels &&
               lhs.arrayLayers == rhs.arrayLayers &&
               lhs.cubeCompatible == rhs.cubeCompatible &&
               lhs.format == rhs.format &&
               lhs.usage == rhs.usage &&
               lhs.memoryUsage == rhs.memoryUsage;
    }

    bool DX12GraphResourceRegistry::bufferDescMatches(
        const BufferDesc& lhs,
        const BufferDesc& rhs) noexcept
    {
        return lhs.size == rhs.size &&
               lhs.usage == rhs.usage &&
               lhs.memoryUsage == rhs.memoryUsage &&
               lhs.mappedAtCreation == rhs.mappedAtCreation;
    }

    BufferDesc DX12GraphResourceRegistry::resolvedBufferDesc(
        const GraphResource& resource) const noexcept
    {
        BufferDesc desc = resource.bufferDesc;
        if (resource.semantic ==
                GraphResourceSemantic::GpuDrivenIndirectArguments &&
            desc.elementCount > 0 && m_indirectCommandStride > 0)
        {
            desc.size = static_cast<uint64_t>(desc.elementCount) *
                m_indirectCommandStride;
        }
        return desc;
    }

    bool DX12GraphResourceRegistry::entryMatches(
        const DX12GraphResourceEntry& entry,
        const GraphResource& resource,
        const TextureDesc& resolvedTexture) const noexcept
    {
        if (resource.type == GraphResourceType::Texture)
        {
            return static_cast<bool>(entry.texture) &&
                   textureDescMatches(entry.textureDesc, resolvedTexture);
        }

        return static_cast<bool>(entry.buffer) &&
               bufferDescMatches(entry.bufferDesc, resolvedBufferDesc(resource));
    }

    void DX12GraphResourceRegistry::materializeTransient(
        DX12GraphResourceEntry& entry,
        const GraphResource& resource,
        const TextureDesc& resolvedTexture)
    {
        if (resource.type == GraphResourceType::Texture)
        {
            entry.textureDesc = resolvedTexture;
            entry.texture = m_resourceAllocator->createTexture(resolvedTexture);
            entry.state = entry.texture.initialState;
            createTextureViews(entry);
            return;
        }

        const BufferDesc bufferDesc = resolvedBufferDesc(resource);
        entry.bufferDesc = bufferDesc;
        entry.buffer = m_resourceAllocator->createBuffer(bufferDesc);
        entry.state = entry.buffer.initialState;
        spdlog::info(
            "[DX12GraphResourceRegistry] Stored graph buffer '{}' id={} valid={} size={} usage={} memory={} mapped={}",
            resource.bufferDesc.debugName,
            static_cast<uint32_t>(resource.id),
            static_cast<bool>(entry.buffer),
            entry.buffer.size,
            static_cast<uint32_t>(entry.buffer.usage),
            static_cast<uint32_t>(entry.buffer.memoryUsage),
            entry.buffer.mappedAtCreation);
    }

    void DX12GraphResourceRegistry::createTextureViews(
        DX12GraphResourceEntry& entry)
    {
        const TextureDesc& desc = entry.textureDesc;

        if (hasFlag(desc.usage, TextureUsageFlags::ColorAttachment))
        {
            entry.rtv = m_descriptorSystem->allocateRTV(1);
            D3D12_RENDER_TARGET_VIEW_DESC rtv{};
            rtv.Format = entry.texture.desc.Format;
            rtv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
            m_device->CreateRenderTargetView(
                entry.texture.resource.Get(), &rtv, entry.rtv.cpuStart);
        }
        if (hasFlag(desc.usage, TextureUsageFlags::DepthAttachment))
        {
            entry.dsv = m_descriptorSystem->allocateDSV(1);
            D3D12_DEPTH_STENCIL_VIEW_DESC dsv{};
            dsv.Format = DXGI_FORMAT_D32_FLOAT;
            dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
            m_device->CreateDepthStencilView(
                entry.texture.resource.Get(), &dsv, entry.dsv.cpuStart);
        }
        if (hasFlag(desc.usage, TextureUsageFlags::Sampled))
        {
            entry.mipSrvs.resize(desc.mipLevels);
            for (uint32_t mip = 0; mip < desc.mipLevels; ++mip)
            {
                entry.mipSrvs[mip] =
                    m_descriptorSystem->allocateResourceDescriptors(1);
                D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
                srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                srv.Format = desc.format == TextureFormat::D32_Float
                    ? DXGI_FORMAT_R32_FLOAT
                    : entry.texture.desc.Format;
                // Single-channel formats (depth-as-color, Hi-Z pyramid) only
                // carry data in R; replicate it into G/B so consumers that
                // read all three (e.g. the ImGui debug view) see greyscale
                // instead of a red tint from unspecified G/B. Multi-channel
                // formats keep the identity mapping.
                srv.Shader4ComponentMapping =
                    srv.Format == DXGI_FORMAT_R32_FLOAT
                        ? D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(
                            0, 0, 0,
                            D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_1)
                        : D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                srv.Texture2D.MostDetailedMip = mip;
                srv.Texture2D.MipLevels = 1;
                m_device->CreateShaderResourceView(
                    entry.texture.resource.Get(),
                    &srv,
                    entry.mipSrvs[mip].cpuStart);
            }
        }
        if (hasFlag(desc.usage, TextureUsageFlags::Storage))
        {
            entry.mipUavs.resize(desc.mipLevels);
            for (uint32_t mip = 0; mip < desc.mipLevels; ++mip)
            {
                entry.mipUavs[mip] =
                    m_descriptorSystem->allocateResourceDescriptors(1);
                D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
                uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
                uav.Format = entry.texture.desc.Format;
                uav.Texture2D.MipSlice = mip;
                m_device->CreateUnorderedAccessView(
                    entry.texture.resource.Get(),
                    nullptr,
                    &uav,
                    entry.mipUavs[mip].cpuStart);
            }
        }
    }

    void DX12GraphResourceRegistry::updateImported(
        DX12GraphResourceEntry& entry,
        const GraphResource& resource,
        const TextureDesc& resolvedTexture,
        const DX12GraphResourceImports& imports) const noexcept
    {
        entry.id = resource.id;
        entry.type = resource.type;
        entry.ownership = resource.ownership;
        entry.imported = resource.imported;
        entry.textureDesc = resolvedTexture;

        switch (resource.imported)
        {
        case ImportedResource::Swapchain:
            entry.importedResource = imports.swapchainResource;
            entry.importedRtv = imports.swapchainRtv;
            break;
        case ImportedResource::None:
            entry.importedResource = nullptr;
            entry.importedRtv = {};
            break;
        }
    }

    void DX12GraphResourceRegistry::retireEntry(
        DX12GraphResourceEntry&& entry,
        uint32_t frameSlot)
    {
        // Nothing owned means nothing to defer.
        if (!entry.texture && !entry.buffer && !entry.rtv.valid() &&
            !entry.dsv.valid() && entry.mipSrvs.empty() &&
            entry.mipUavs.empty())
        {
            return;
        }

        assert(frameSlot < m_retiredByFrameSlot.size());
        if (frameSlot >= m_retiredByFrameSlot.size())
        {
            // Out-of-range in a Release build (assert compiled out): destroy
            // immediately rather than indexing out of bounds. This forgoes
            // the deferred-free-until-slot-recycled guarantee, but that is
            // strictly safer than corrupting the heap.
            destroyEntry(entry);
            return;
        }
        m_retiredByFrameSlot[frameSlot].push_back(std::move(entry));
    }

    void DX12GraphResourceRegistry::destroyEntry(
        DX12GraphResourceEntry& entry) noexcept
    {
        if (m_resourceAllocator)
        {
            m_resourceAllocator->destroyTexture(entry.texture);
            m_resourceAllocator->destroyBuffer(entry.buffer);
        }
        if (m_descriptorSystem)
        {
            m_descriptorSystem->releaseRTV(entry.rtv);
            m_descriptorSystem->releaseDSV(entry.dsv);
            for (DX12DescriptorAllocation& allocation : entry.mipSrvs)
            {
                m_descriptorSystem->releaseResourceDescriptors(allocation);
            }
            for (DX12DescriptorAllocation& allocation : entry.mipUavs)
            {
                m_descriptorSystem->releaseResourceDescriptors(allocation);
            }
        }
        entry.rtv = {};
        entry.dsv = {};
        entry.mipSrvs.clear();
        entry.mipUavs.clear();
    }

    void DX12GraphResourceRegistry::validateFrameSlot(uint32_t frameSlot) const
    {
        assert(frameSlot < m_retiredByFrameSlot.size() &&
            "DX12GraphResourceRegistry::materialize frame slot out of range");
        (void)frameSlot;
    }

    uint64_t DX12GraphResourceRegistry::nextGeneration() noexcept
    {
        return ++m_materializationGeneration;
    }
}
