#pragma once

#include "ic/renderer/dx12_backend/dx12_descriptor_system.h"
#include "ic/renderer/dx12_backend/dx12_resource_allocator.h"
#include "ic/renderer/frame_graph/compiled_graph_plan.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace ic
{
    class DX12Device;

    struct DX12GraphResourceImports
    {
        // Imported resources are borrowed for the duration of the frame. The
        // registry never releases either the resource or its swapchain owned
        // descriptor.
        ID3D12Resource* swapchainResource = nullptr;
        D3D12_CPU_DESCRIPTOR_HANDLE swapchainRtv = {};
    };

    struct DX12GraphResourceEntry
    {
        GraphResourceId id = InvalidGraphResourceId;
        GraphResourceType type = GraphResourceType::Texture;
        ResourceOwnership ownership = ResourceOwnership::Transient;
        ImportedResource imported = ImportedResource::None;

        TextureDesc textureDesc = {};
        BufferDesc bufferDesc = {};

        DX12Texture texture;
        DX12Buffer buffer;

        DX12DescriptorAllocation rtv;
        DX12DescriptorAllocation dsv;
        std::vector<DX12DescriptorAllocation> mipSrvs;
        std::vector<DX12DescriptorAllocation> mipUavs;

        ID3D12Resource* importedResource = nullptr;
        D3D12_CPU_DESCRIPTOR_HANDLE importedRtv = {};

        D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
        GraphNodeId firstUse = InvalidGraphNodeId;
        GraphNodeId lastUse = InvalidGraphNodeId;

        DX12GraphResourceEntry() = default;
        DX12GraphResourceEntry(const DX12GraphResourceEntry&) = delete;
        DX12GraphResourceEntry& operator=(const DX12GraphResourceEntry&) = delete;
        DX12GraphResourceEntry(DX12GraphResourceEntry&&) noexcept = default;
        DX12GraphResourceEntry& operator=(DX12GraphResourceEntry&&) noexcept = default;

        [[nodiscard]] ID3D12Resource* nativeResource() const noexcept
        {
            if (ownership == ResourceOwnership::Imported)
            {
                return importedResource;
            }

            return type == GraphResourceType::Texture
                ? texture.resource.Get()
                : buffer.resource.Get();
        }

        [[nodiscard]] D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle() const noexcept
        {
            return ownership == ResourceOwnership::Imported
                ? importedRtv
                : rtv.cpuStart;
        }

        [[nodiscard]] uint32_t width() const noexcept
        {
            return textureDesc.width;
        }

        [[nodiscard]] uint32_t height() const noexcept
        {
            return textureDesc.height;
        }

        [[nodiscard]] uint32_t mipLevels() const noexcept
        {
            return textureDesc.mipLevels;
        }
    };

    // All physical instances backing one logical graph resource. Single-instance
    // resources (persistent, imported) hold one slot; per-frame-slot and history
    // resources hold one slot per frame in flight so two frames never share the
    // same physical memory. The registry hands out the current frame's slot from
    // entry() and the prior frame's slot from previousEntry().
    struct DX12GraphResourceInstances
    {
        ResourceMultiplicity multiplicity = ResourceMultiplicity::Single;
        std::vector<DX12GraphResourceEntry> slots;
        uint64_t materializationGeneration = 0;
    };

    class DX12GraphResourceRegistry final
    {
    public:
        DX12GraphResourceRegistry() = default;
        DX12GraphResourceRegistry(const DX12GraphResourceRegistry&) = delete;
        DX12GraphResourceRegistry& operator=(
            const DX12GraphResourceRegistry&) = delete;

        void init(
            const DX12Device& device,
            DX12ResourceAllocator& resourceAllocator,
            DX12DescriptorSystem& descriptorSystem,
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
        // entries retain their native resources and descriptor allocations.
        void materialize(
            const CompiledGraphPlan& plan,
            uint32_t frameSlot,
            uint32_t defaultWidth,
            uint32_t defaultHeight,
            const DX12GraphResourceImports& imports);

        // Current frame slot's instance for this resource.
        [[nodiscard]] DX12GraphResourceEntry* entry(
            GraphResourceId id) noexcept;
        [[nodiscard]] const DX12GraphResourceEntry* entry(
            GraphResourceId id) const noexcept;

        // Previous frame slot's instance, for history/ping-pong resources. For
        // single-instance resources this returns the same instance as entry().
        // Returns nullptr if the resource has not been materialized.
        [[nodiscard]] DX12GraphResourceEntry* previousEntry(
            GraphResourceId id) noexcept;
        [[nodiscard]] const DX12GraphResourceEntry* previousEntry(
            GraphResourceId id) const noexcept;

        [[nodiscard]] ID3D12Resource* nativeResource(
            GraphResourceId id) const noexcept;
        [[nodiscard]] D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle(
            GraphResourceId id) const noexcept;

        // Byte stride of one native GPU-driven indirect command on this backend.
        // Set by the backend at init so the registry can size element-count
        // buffers (GpuDrivenIndirectArguments) at their native layout.
        void setNativeIndirectCommandStride(uint32_t stride) noexcept
        {
            m_indirectCommandStride = stride;
        }

        [[nodiscard]] uint32_t frameSlotCount() const noexcept
        {
            return static_cast<uint32_t>(m_retiredByFrameSlot.size());
        }

        [[nodiscard]] size_t size() const noexcept
        {
            return m_entries.size();
        }

    private:
        using EntryMap =
            std::unordered_map<GraphResourceId, DX12GraphResourceInstances>;

        // Ring size for a multiplicity: single instance, or one per frame slot.
        [[nodiscard]] uint32_t instanceCountFor(
            ResourceMultiplicity multiplicity) const noexcept;
        [[nodiscard]] uint32_t currentInstanceIndex(
            const DX12GraphResourceInstances& instances) const noexcept;
        [[nodiscard]] uint32_t previousInstanceIndex(
            const DX12GraphResourceInstances& instances) const noexcept;

        static TextureDesc resolvedTextureDesc(
            const TextureDesc& desc,
            uint32_t defaultWidth,
            uint32_t defaultHeight) noexcept;
        static bool textureDescMatches(
            const TextureDesc& lhs,
            const TextureDesc& rhs) noexcept;
        static bool bufferDescMatches(
            const BufferDesc& lhs,
            const BufferDesc& rhs) noexcept;

        // Resolves a buffer's effective byte size: for a native-stride semantic
        // (GpuDrivenIndirectArguments) declared by element count, size becomes
        // elementCount * m_indirectCommandStride; otherwise desc.size stands.
        [[nodiscard]] BufferDesc resolvedBufferDesc(
            const GraphResource& resource) const noexcept;

        [[nodiscard]] bool entryMatches(
            const DX12GraphResourceEntry& entry,
            const GraphResource& resource,
            const TextureDesc& resolvedTexture) const noexcept;

        void materializeTransient(
            DX12GraphResourceEntry& entry,
            const GraphResource& resource,
            const TextureDesc& resolvedTexture);
        void updateImported(
            DX12GraphResourceEntry& entry,
            const GraphResource& resource,
            const TextureDesc& resolvedTexture,
            const DX12GraphResourceImports& imports) const noexcept;
        void createTextureViews(DX12GraphResourceEntry& entry);

        void retireEntry(
            DX12GraphResourceEntry&& entry,
            uint32_t frameSlot);
        void destroyEntry(DX12GraphResourceEntry& entry) noexcept;
        void validateFrameSlot(uint32_t frameSlot) const;
        uint64_t nextGeneration() noexcept;

        ID3D12Device5* m_device = nullptr;
        DX12ResourceAllocator* m_resourceAllocator = nullptr;
        DX12DescriptorSystem* m_descriptorSystem = nullptr;

        EntryMap m_entries;
        std::vector<std::vector<DX12GraphResourceEntry>>
            m_retiredByFrameSlot;
        uint64_t m_materializationGeneration = 0;
        uint32_t m_framesInFlight = 1;
        // Native byte stride of one GPU-driven indirect command on this backend
        // (DX12: sizeof(DX12GpuIndexedIndirectCommand)).
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
