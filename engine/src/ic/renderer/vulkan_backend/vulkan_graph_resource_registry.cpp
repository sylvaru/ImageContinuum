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
        m_retiredByFrameSlot.clear();
        m_retiredByFrameSlot.resize(std::max(1u, framesInFlight));
        m_contentGeneration = 0;
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
        for (auto& [id, entry] : m_entries)
        {
            destroyEntry(entry);
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

        for (const GraphResource& resource : plan.resources)
        {
            VulkanGraphResourceEntry& entry = m_entries[resource.id];

            uint32_t resolvedWidth = resource.textureDesc.width;
            uint32_t resolvedHeight = resource.textureDesc.height;
            if (resolvedWidth == 0) resolvedWidth = defaultWidth;
            if (resolvedHeight == 0) resolvedHeight = defaultHeight;

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
    }

    VulkanGraphResourceEntry* VulkanGraphResourceRegistry::entry(
        GraphResourceId id) noexcept
    {
        auto it = m_entries.find(id);
        return it != m_entries.end() ? &it->second : nullptr;
    }

    const VulkanGraphResourceEntry* VulkanGraphResourceRegistry::entry(
        GraphResourceId id) const noexcept
    {
        auto it = m_entries.find(id);
        return it != m_entries.end() ? &it->second : nullptr;
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
                   entry.width == resolvedWidth &&
                   entry.height == resolvedHeight &&
                   entry.mipLevels == resource.textureDesc.mipLevels &&
                   entry.arrayLayers == resource.textureDesc.arrayLayers;
        }

        return static_cast<bool>(entry.buffer) &&
               entry.buffer.size == resource.bufferDesc.size;
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
        entry.buffer = m_resourceAllocator->createBuffer(resource.bufferDesc);
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
