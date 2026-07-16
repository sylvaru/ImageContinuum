#pragma once

#include "ic/renderer/vulkan_backend/vulkan_resource_allocator.h"
#include "ic/renderer/frame_graph/compiled_graph_plan.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

namespace ic
{
    class VulkanDevice;

    struct VulkanGraphResourceImports
    {
        // Imported resources are borrowed for the duration of the frame. The
        // registry never releases the swapchain image (owned by the swapchain).
        VkImage swapchainImage = VK_NULL_HANDLE;
    };

    struct VulkanGraphResourceEntry
    {
        GraphResourceId id = InvalidGraphResourceId;
        GraphResourceType type = GraphResourceType::Texture;
        ResourceOwnership ownership = ResourceOwnership::Transient;
        ImportedResource imported = ImportedResource::None;

        VulkanTexture texture;
        VulkanBuffer buffer;
        TextureDesc textureDesc = {};
        BufferDesc bufferDesc = {};
        VkImageView view = VK_NULL_HANDLE;
        std::vector<VkImageView> mipViews;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;

        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t mipLevels = 1;
        uint32_t arrayLayers = 1;

        // Bumped each time the native image/buffer is recreated so consumers
        // that cache views into it (e.g. the Hi-Z debug overlay) can detect a
        // recreation and rebuild.
        uint64_t generation = 0;

        VulkanGraphResourceEntry() = default;
        VulkanGraphResourceEntry(const VulkanGraphResourceEntry&) = delete;
        VulkanGraphResourceEntry& operator=(
            const VulkanGraphResourceEntry&) = delete;
        VulkanGraphResourceEntry(
            VulkanGraphResourceEntry&&) noexcept = default;
        VulkanGraphResourceEntry& operator=(
            VulkanGraphResourceEntry&&) noexcept = default;
    };

    // All physical instances backing one logical graph resource. Single-instance
    // resources (persistent, imported) hold one slot; per-frame-slot and history
    // resources hold one slot per frame in flight so two frames never share the
    // same physical memory. entry() hands out the current frame's slot,
    // previousEntry() the prior frame's slot.
    struct VulkanGraphResourceInstances
    {
        ResourceMultiplicity multiplicity = ResourceMultiplicity::Single;
        std::vector<VulkanGraphResourceEntry> slots;
    };

    class VulkanGraphResourceRegistry final
    {
    public:
        VulkanGraphResourceRegistry() = default;
        VulkanGraphResourceRegistry(
            const VulkanGraphResourceRegistry&) = delete;
        VulkanGraphResourceRegistry& operator=(
            const VulkanGraphResourceRegistry&) = delete;

        void init(
            const VulkanDevice& device,
            VulkanResourceAllocator& resourceAllocator,
            uint32_t framesInFlight);

        // The caller must ensure the GPU is idle before shutdown.
        void shutdown();

        // Frees every live and retired resource immediately while keeping the
        // registry usable. The caller must ensure the GPU is idle first (used
        // on swapchain recreation).
        void reset() noexcept;

        // Call only after the executor has waited for this frame slot. Native
        // resources replaced the previous time this slot was used can then be
        // released without adding a GPU wait.
        void recycleFrameSlot(uint32_t frameSlot);

        // Materialization is a serial, pre-recording operation. Unchanged
        // entries retain their native resources and image views.
        void materialize(
            const CompiledGraphPlan& plan,
            uint32_t frameSlot,
            uint32_t defaultWidth,
            uint32_t defaultHeight,
            const VulkanGraphResourceImports& imports);

        // Current frame slot's instance for this resource.
        [[nodiscard]] VulkanGraphResourceEntry* entry(
            GraphResourceId id) noexcept;
        [[nodiscard]] const VulkanGraphResourceEntry* entry(
            GraphResourceId id) const noexcept;

        // Previous frame slot's instance, for history/ping-pong resources. For
        // single-instance resources this returns the same instance as entry().
        // Returns nullptr if the resource has not been materialized.
        [[nodiscard]] VulkanGraphResourceEntry* previousEntry(
            GraphResourceId id) noexcept;
        [[nodiscard]] const VulkanGraphResourceEntry* previousEntry(
            GraphResourceId id) const noexcept;

        // Byte stride of one native GPU-driven indirect command on this backend
        // (Vulkan: sizeof(VkDrawIndexedIndirectCommand)). Set by the backend at
        // init so element-count buffers (GpuDrivenIndirectArguments) are sized
        // at their native layout.
        void setNativeIndirectCommandStride(uint32_t stride) noexcept
        {
            m_indirectCommandStride = stride;
        }

        [[nodiscard]] size_t size() const noexcept
        {
            return m_entries.size();
        }

    private:

        // Resolves a buffer's effective byte size: for a native-stride semantic
        // (GpuDrivenIndirectArguments) declared by element count, size becomes
        // elementCount * m_indirectCommandStride; otherwise desc.size stands.
        [[nodiscard]] BufferDesc resolvedBufferDesc(
            const GraphResource& resource) const noexcept;
        using EntryMap =
            std::unordered_map<GraphResourceId, VulkanGraphResourceInstances>;

        [[nodiscard]] uint32_t instanceCountFor(
            ResourceMultiplicity multiplicity) const noexcept;
        [[nodiscard]] uint32_t currentInstanceIndex(
            const VulkanGraphResourceInstances& instances) const noexcept;
        [[nodiscard]] uint32_t previousInstanceIndex(
            const VulkanGraphResourceInstances& instances) const noexcept;

        [[nodiscard]] bool entryMatches(
            const VulkanGraphResourceEntry& entry,
            const GraphResource& resource,
            uint32_t resolvedWidth,
            uint32_t resolvedHeight) const noexcept;

        void materializeTexture(
            VulkanGraphResourceEntry& entry,
            const GraphResource& resource,
            uint32_t resolvedWidth,
            uint32_t resolvedHeight);
        void materializeBuffer(
            VulkanGraphResourceEntry& entry,
            const GraphResource& resource);
        void updateImported(
            VulkanGraphResourceEntry& entry,
            const GraphResource& resource,
            uint32_t resolvedWidth,
            uint32_t resolvedHeight,
            const VulkanGraphResourceImports& imports) const noexcept;

        void retireEntry(
            VulkanGraphResourceEntry&& entry,
            uint32_t frameSlot);
        void destroyEntry(VulkanGraphResourceEntry& entry) noexcept;
        uint64_t nextGeneration() noexcept;

        VkDevice m_device = VK_NULL_HANDLE;
        VulkanResourceAllocator* m_resourceAllocator = nullptr;

        EntryMap m_entries;
        std::vector<std::vector<VulkanGraphResourceEntry>> m_retiredByFrameSlot;
        uint64_t m_contentGeneration = 0;
        uint32_t m_framesInFlight = 1;
        // Native byte stride of one GPU-driven indirect command on this backend
        // (Vulkan: sizeof(VkDrawIndexedIndirectCommand)).
        uint32_t m_indirectCommandStride = 0;
        // Executor frame slot for the current frame: drives per-frame-slot
        // instance selection so it aligns with frame-slot-indexed backend state.
        uint32_t m_frameSlot = 0;
        // Monotonic count of submitted frames (materialize calls): drives
        // history current/previous selection. Independent of frameIndex so
        // skipped frames never advance the history ring.
        uint64_t m_frameCounter = 0;
    };
}
