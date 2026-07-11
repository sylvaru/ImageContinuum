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
        m_retiredByFrameSlot.clear();
        m_retiredByFrameSlot.resize(std::max(1u, framesInFlight));
        m_materializationGeneration = 0;
    }

    void DX12GraphResourceRegistry::shutdown()
    {
        for (auto& [id, entry] : m_entries)
        {
            destroyEntry(entry);
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
        for (auto& [id, entry] : m_entries)
        {
            destroyEntry(entry);
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
        const uint64_t generation = nextGeneration();

        for (const GraphResource& resource : plan.resources)
        {
            DX12GraphResourceEntry& entry = m_entries[resource.id];
            entry.materializationGeneration = generation;

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
                // state exactly as they were on the previous frame.
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
            entry.materializationGeneration = generation;
            entry.type = resource.type;
            entry.ownership = resource.ownership;
            entry.imported = resource.imported;

            materializeTransient(entry, resource, resolvedTexture);
        }
    }

    DX12GraphResourceEntry* DX12GraphResourceRegistry::entry(
        GraphResourceId id) noexcept
    {
        auto it = m_entries.find(id);
        return it != m_entries.end() ? &it->second : nullptr;
    }

    const DX12GraphResourceEntry* DX12GraphResourceRegistry::entry(
        GraphResourceId id) const noexcept
    {
        auto it = m_entries.find(id);
        return it != m_entries.end() ? &it->second : nullptr;
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
               lhs.mipLevels == rhs.mipLevels &&
               lhs.arrayLayers == rhs.arrayLayers;
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
               bufferDescMatches(entry.bufferDesc, resource.bufferDesc);
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

        entry.bufferDesc = resource.bufferDesc;
        entry.buffer = m_resourceAllocator->createBuffer(resource.bufferDesc);
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
