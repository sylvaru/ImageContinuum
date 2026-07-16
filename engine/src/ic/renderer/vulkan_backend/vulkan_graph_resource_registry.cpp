#include "ic/renderer/vulkan_backend/vulkan_graph_resource_registry.h"

#include "ic/renderer/vulkan_backend/vulkan_device.h"
#include "ic/renderer/render_types.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <utility>

namespace ic
{
    namespace
    {
        void throwIfViewFailed(VkResult result, const char* message)
        {
            if (result != VK_SUCCESS)
            {
                throw std::runtime_error(message);
            }
        }
    }

    void VulkanGraphResourceRegistry::init(
        const VulkanDevice& device,
        VulkanResourceAllocator& resourceAllocator,
        uint32_t framesInFlight)
    {
        m_device = device.device();
        m_resourceAllocator = &resourceAllocator;
        m_framesInFlight = std::max(1u, framesInFlight);
        m_frameCounter = 0;
        m_retiredByFrameSlot.clear();
        m_retiredByFrameSlot.resize(m_framesInFlight);
        m_contentGeneration = 0;
    }

    uint32_t VulkanGraphResourceRegistry::instanceCountFor(
        ResourceMultiplicity multiplicity) const noexcept
    {
        return multiplicity == ResourceMultiplicity::Single
            ? 1u
            : m_framesInFlight;
    }

    uint32_t VulkanGraphResourceRegistry::currentInstanceIndex(
        const VulkanGraphResourceInstances& instances) const noexcept
    {
        const uint32_t count = static_cast<uint32_t>(instances.slots.size());
        // Per-frame-slot resources index by the executor's frame slot so they
        // align with every other frame-slot-indexed backend structure (command
        // buffers, cached descriptor sets, retirement buckets). Only history
        // resources index by the monotonic submitted-frame counter. That is
        // what makes their "previous" skip-safe.
        if (instances.multiplicity == ResourceMultiplicity::History)
        {
            return currentFrameInstanceIndex(m_frameCounter, count);
        }
        return currentFrameInstanceIndex(m_frameSlot, count);
    }

    uint32_t VulkanGraphResourceRegistry::previousInstanceIndex(
        const VulkanGraphResourceInstances& instances) const noexcept
    {
        return previousFrameInstanceIndex(
            m_frameCounter, static_cast<uint32_t>(instances.slots.size()));
    }

    void VulkanGraphResourceRegistry::shutdown()
    {
        reset();
        m_retiredByFrameSlot.clear();
        m_device = VK_NULL_HANDLE;
        m_resourceAllocator = nullptr;
    }

    void VulkanGraphResourceRegistry::reset() noexcept
    {
        for (auto& [id, instances] : m_entries)
        {
            for (VulkanGraphResourceEntry& entry : instances.slots)
            {
                destroyEntry(entry);
            }
        }
        m_entries.clear();

        for (std::vector<VulkanGraphResourceEntry>& retired :
             m_retiredByFrameSlot)
        {
            for (VulkanGraphResourceEntry& entry : retired)
            {
                destroyEntry(entry);
            }
            retired.clear();
        }
    }

    void VulkanGraphResourceRegistry::recycleFrameSlot(uint32_t frameSlot)
    {
        if (frameSlot >= m_retiredByFrameSlot.size())
        {
            return;
        }

        std::vector<VulkanGraphResourceEntry>& retired =
            m_retiredByFrameSlot[frameSlot];
        for (VulkanGraphResourceEntry& entry : retired)
        {
            destroyEntry(entry);
        }
        retired.clear();
    }

    void VulkanGraphResourceRegistry::materialize(
        const CompiledGraphPlan& plan,
        uint32_t frameSlot,
        uint32_t defaultWidth,
        uint32_t defaultHeight,
        const VulkanGraphResourceImports& imports)
    {
        assert(frameSlot < m_retiredByFrameSlot.size());
        // Executor frame slot (drives per-frame-slot instance selection).
        m_frameSlot = frameSlot;
        // Monotonic submitted-frame count (drives history current/previous
        // selection), advanced once per recorded frame regardless of skips.
        ++m_frameCounter;

        for (const GraphResource& resource : plan.resources)
        {
            VulkanGraphResourceInstances& instances = m_entries[resource.id];
            instances.multiplicity = resource.multiplicity;

            uint32_t resolvedWidth = resource.textureDesc.width;
            uint32_t resolvedHeight = resource.textureDesc.height;
            if (resolvedWidth == 0) resolvedWidth = defaultWidth;
            if (resolvedHeight == 0) resolvedHeight = defaultHeight;

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
            VulkanGraphResourceEntry& entry = instances.slots[index];

            if (resource.ownership == ResourceOwnership::Imported)
            {
                updateImported(
                    entry, resource, resolvedWidth, resolvedHeight, imports);
                continue;
            }

            if (entryMatches(entry, resource, resolvedWidth, resolvedHeight))
            {
                // Unchanged: keep native resource, views and tracked layout.
                entry.id = resource.id;
                entry.type = resource.type;
                entry.ownership = resource.ownership;
                entry.imported = resource.imported;
                continue;
            }

            // Recreation required. Retire the old native contents into the
            // frame slot bucket so they are freed only after the executor has
            // waited this slot
            retireEntry(
                std::exchange(entry, VulkanGraphResourceEntry{}), frameSlot);

            entry.id = resource.id;
            entry.type = resource.type;
            entry.ownership = resource.ownership;
            entry.imported = resource.imported;

            if (resource.type == GraphResourceType::Texture)
            {
                materializeTexture(
                    entry, resource, resolvedWidth, resolvedHeight);
            }
            else
            {
                materializeBuffer(entry, resource);
            }
        }

        // Plans may shrink or replace their resource set. Retire entries that
        // are no longer described by this plan instead of retaining native
        // allocations and descriptors indefinitely.
        for (auto it = m_entries.begin(); it != m_entries.end();)
        {
            const GraphResourceId id = it->first;
            const bool active = id < plan.resources.size() &&
                plan.resources[id].id == id;
            if (active)
            {
                ++it;
                continue;
            }

            for (VulkanGraphResourceEntry& entry : it->second.slots)
            {
                retireEntry(std::move(entry), frameSlot);
            }
            it = m_entries.erase(it);
        }
    }

    VulkanGraphResourceEntry* VulkanGraphResourceRegistry::entry(
        GraphResourceId id) noexcept
    {
        auto it = m_entries.find(id);
        if (it == m_entries.end() || it->second.slots.empty())
        {
            return nullptr;
        }
        return &it->second.slots[currentInstanceIndex(it->second)];
    }

    const VulkanGraphResourceEntry* VulkanGraphResourceRegistry::entry(
        GraphResourceId id) const noexcept
    {
        auto it = m_entries.find(id);
        if (it == m_entries.end() || it->second.slots.empty())
        {
            return nullptr;
        }
        return &it->second.slots[currentInstanceIndex(it->second)];
    }

    VulkanGraphResourceEntry* VulkanGraphResourceRegistry::previousEntry(
        GraphResourceId id) noexcept
    {
        auto it = m_entries.find(id);
        if (it == m_entries.end() || it->second.slots.empty())
        {
            return nullptr;
        }
        return &it->second.slots[previousInstanceIndex(it->second)];
    }

    const VulkanGraphResourceEntry* VulkanGraphResourceRegistry::previousEntry(
        GraphResourceId id) const noexcept
    {
        auto it = m_entries.find(id);
        if (it == m_entries.end() || it->second.slots.empty())
        {
            return nullptr;
        }
        return &it->second.slots[previousInstanceIndex(it->second)];
    }

    bool VulkanGraphResourceRegistry::entryMatches(
        const VulkanGraphResourceEntry& entry,
        const GraphResource& resource,
        uint32_t resolvedWidth,
        uint32_t resolvedHeight) const noexcept
    {
        if (resource.type == GraphResourceType::Texture)
        {
            return static_cast<bool>(entry.texture) &&
                   entry.textureDesc.width == resolvedWidth &&
                   entry.textureDesc.height == resolvedHeight &&
                   entry.textureDesc.depth == resource.textureDesc.depth &&
                   entry.textureDesc.mipLevels == resource.textureDesc.mipLevels &&
                   entry.textureDesc.arrayLayers == resource.textureDesc.arrayLayers &&
                   entry.textureDesc.cubeCompatible == resource.textureDesc.cubeCompatible &&
                   entry.textureDesc.format == resource.textureDesc.format &&
                   entry.textureDesc.usage == resource.textureDesc.usage &&
                   entry.textureDesc.memoryUsage == resource.textureDesc.memoryUsage;
        }

        const BufferDesc desc = resolvedBufferDesc(resource);
        return static_cast<bool>(entry.buffer) &&
               entry.bufferDesc.size == desc.size &&
               entry.bufferDesc.usage == desc.usage &&
               entry.bufferDesc.memoryUsage == desc.memoryUsage &&
               entry.bufferDesc.mappedAtCreation == desc.mappedAtCreation;
    }

    BufferDesc VulkanGraphResourceRegistry::resolvedBufferDesc(
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

    void VulkanGraphResourceRegistry::materializeTexture(
        VulkanGraphResourceEntry& entry,
        const GraphResource& resource,
        uint32_t resolvedWidth,
        uint32_t resolvedHeight)
    {
        TextureDesc desc = resource.textureDesc;
        desc.width = resolvedWidth;
        desc.height = resolvedHeight;
        entry.textureDesc = desc;

        entry.texture = m_resourceAllocator->createTexture(desc);
        entry.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        entry.width = desc.width;
        entry.height = desc.height;
        entry.mipLevels = desc.mipLevels;
        entry.arrayLayers = desc.arrayLayers;
        entry.generation = nextGeneration();

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = entry.texture.image;
        viewInfo.viewType = desc.arrayLayers > 1
            ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
            : VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = entry.texture.format;
        viewInfo.subresourceRange.aspectMask =
            hasFlag(desc.usage, TextureUsageFlags::DepthAttachment)
                ? VK_IMAGE_ASPECT_DEPTH_BIT
                : VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.levelCount = desc.mipLevels;
        viewInfo.subresourceRange.layerCount = desc.arrayLayers;
        throwIfViewFailed(
            vkCreateImageView(m_device, &viewInfo, nullptr, &entry.view),
            "Failed to create Vulkan graph texture view.");

        if (desc.mipLevels > 1)
        {
            entry.mipViews.resize(desc.mipLevels);
            for (uint32_t mip = 0; mip < desc.mipLevels; ++mip)
            {
                viewInfo.subresourceRange.baseMipLevel = mip;
                viewInfo.subresourceRange.levelCount = 1;
                throwIfViewFailed(
                    vkCreateImageView(
                        m_device, &viewInfo, nullptr, &entry.mipViews[mip]),
                    "Failed to create Vulkan graph texture mip view.");
            }
        }
    }

    void VulkanGraphResourceRegistry::materializeBuffer(
        VulkanGraphResourceEntry& entry,
        const GraphResource& resource)
    {
        const BufferDesc bufferDesc = resolvedBufferDesc(resource);
        entry.bufferDesc = bufferDesc;
        entry.buffer = m_resourceAllocator->createBuffer(bufferDesc);
        entry.generation = nextGeneration();
    }

    void VulkanGraphResourceRegistry::updateImported(
        VulkanGraphResourceEntry& entry,
        const GraphResource& resource,
        uint32_t resolvedWidth,
        uint32_t resolvedHeight,
        const VulkanGraphResourceImports& imports) const noexcept
    {
        entry.id = resource.id;
        entry.type = resource.type;
        entry.ownership = resource.ownership;
        entry.imported = resource.imported;
        entry.width = resolvedWidth;
        entry.height = resolvedHeight;
        (void)imports;
    }

    void VulkanGraphResourceRegistry::retireEntry(
        VulkanGraphResourceEntry&& entry,
        uint32_t frameSlot)
    {
        if (!entry.texture && !entry.buffer &&
            entry.view == VK_NULL_HANDLE && entry.mipViews.empty())
        {
            return;
        }

        assert(frameSlot < m_retiredByFrameSlot.size());
        m_retiredByFrameSlot[frameSlot].push_back(std::move(entry));
    }

    void VulkanGraphResourceRegistry::destroyEntry(
        VulkanGraphResourceEntry& entry) noexcept
    {
        if (m_device != VK_NULL_HANDLE)
        {
            if (entry.view != VK_NULL_HANDLE)
            {
                vkDestroyImageView(m_device, entry.view, nullptr);
                entry.view = VK_NULL_HANDLE;
            }
            for (VkImageView mipView : entry.mipViews)
            {
                if (mipView != VK_NULL_HANDLE)
                {
                    vkDestroyImageView(m_device, mipView, nullptr);
                }
            }
        }
        entry.mipViews.clear();

        if (m_resourceAllocator)
        {
            m_resourceAllocator->destroyTexture(entry.texture);
            m_resourceAllocator->destroyBuffer(entry.buffer);
        }
    }

    uint64_t VulkanGraphResourceRegistry::nextGeneration() noexcept
    {
        return ++m_contentGeneration;
    }
}
