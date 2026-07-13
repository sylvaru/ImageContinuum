#pragma once

#include "ic/renderer/dx12_backend/dx12_descriptor_system.h"
#include "ic/renderer/dx12_backend/dx12_resource_allocator.h"
#include "ic/renderer/renderer_gpu_assets.h"
#include "ic/renderer/gpu_scene_preparation.h"
#include "ic/core/asset_manager.h"
#include "ic/scene/scene_render_view.h"

#include <cstdint>
#include <functional>
#include <span>
#include <unordered_map>
#include <vector>

#include <d3d12.h>
#include <wrl/client.h>

namespace ic
{
    class DX12Device;

    // GPU-resident representation of an uploaded model. Shared vocabulary
    // between the backend's asset-upload cache (which owns residency) and the
    // GPU scene's per-frame draw preparation (which only references it).
    struct DX12UploadedModel
    {
        DX12Buffer vertexBuffer;
        DX12Buffer indexBuffer;
        std::vector<GpuMesh> meshes;
        std::vector<glm::mat4> meshTransforms;
        std::vector<GpuMaterialData> materials;
        std::vector<uint32_t> textureDescriptorIndices;
        std::vector<uint32_t> samplerDescriptorIndices;
        bool uploaded = false;
    };

    // Persistent, growth-only per-frame-slot scene upload buffers.
    struct DX12GpuSceneFrameResources
    {
        DX12Buffer frameConstants;
        DX12Buffer objects;
        DX12Buffer materials;
        DX12Buffer visibleLights;
        DX12Buffer instanceBounds;
        DX12Buffer drawInputs;

        DX12DescriptorAllocation objectSrv;
        DX12DescriptorAllocation materialSrv;

        uint32_t objectCapacity = 0;
        uint32_t materialCapacity = 0;
        uint32_t visibleLightCapacity = 0;
        uint32_t instanceBoundsCapacity = 0;
        uint32_t drawInputCapacity = 0;
    };

    // Owns the persistent GPU-driven scene: CPU-side draw extraction/sorting/
    // binning (via the API-neutral PreparedGpuScene), the growth-only per-frame
    // upload buffers, and the GPU-driven cull/indirect-argument buffers
    // consumed by the frustum-cull compute pass and (once activated)
    // ExecuteIndirect. The cull/indirect buffers hold native D3D12 formats
    // (DX12GpuIndexedIndirectCommand); the frustum-cull compute shader writes
    // them directly. DX12 prefixes each native draw with root constants so an
    // ExecuteIndirect command identifies its object/material without relying
    // on Vulkan's firstInstance/SV_InstanceID semantics.
    struct DX12GpuIndexedIndirectCommand
    {
        DrawConstants drawConstants{};
        GpuIndexedIndirectArguments draw{};
    };
    static_assert(sizeof(DX12GpuIndexedIndirectCommand) == 36);

    class DX12GpuScene final
    {
    public:
        using DrawItem = PreparedGpuDraw;
        using GeometryBin = PreparedGpuGeometryBin;

        // Maps a scene item's AssetHandle to its uploaded GPU model view for
        // CPU-side extraction (spans borrowed from the caller's residency
        // storage). Residency is owned by the caller, not the GpuScene.
        using ResolveModelFn = std::function<GpuSceneModelView(AssetHandle)>;
        using BuildFrameDataFn = std::function<GpuFrameData(
            uint32_t visibleLightCount,
            uint32_t instanceBoundsCount,
            uint32_t geometryBinCount)>;

        DX12GpuScene() = default;
        DX12GpuScene(const DX12GpuScene&) = delete;
        DX12GpuScene& operator=(const DX12GpuScene&) = delete;

        void init(
            const DX12Device& device,
            DX12ResourceAllocator& resourceAllocator,
            DX12DescriptorSystem& descriptorSystem,
            uint32_t framesInFlight);
        void shutdown();

        // Extracts, sorts, bins and uploads this frame's scene draw data.
        // Cached per frameIndex: a second call for the same frame skips
        // re-extraction and re-upload. `buildFrameData` is invoked only on a
        // fresh extraction (so it can read final instance/geometry-bin
        // counts) and is where the caller should ensure any renderer-specific
        // per-resolution resources (e.g. clustered-forward grids) before
        // composing GpuFrameData. Returns false when there is nothing to draw.
        bool prepare(
            uint64_t frameIndex,
            const SceneRenderView& scene,
            uint32_t frameSlot,
            const ResolveModelFn& resolveModel,
            const BuildFrameDataFn& buildFrameData);

        [[nodiscard]] bool valid() const noexcept { return m_prepared.valid(); }
        [[nodiscard]] std::span<const DrawItem> draws() const noexcept
        {
            return m_prepared.draws();
        }
        [[nodiscard]] std::span<const GeometryBin> geometryBins() const noexcept
        {
            return m_prepared.geometryBins();
        }
        [[nodiscard]] uint32_t instanceCount() const noexcept
        {
            return static_cast<uint32_t>(m_prepared.instanceBounds().size());
        }

        [[nodiscard]] DX12GpuSceneFrameResources& frameResources(
            uint32_t frameSlot) noexcept
        {
            return m_frames[frameSlot];
        }
        [[nodiscard]] const DX12GpuSceneFrameResources& frameResources(
            uint32_t frameSlot) const noexcept
        {
            return m_frames[frameSlot];
        }
        [[nodiscard]] uint32_t frameSlotCount() const noexcept
        {
            return static_cast<uint32_t>(m_frames.size());
        }

        // Ensures the GPU-driven cull/indirect buffers exist for the given
        // capacity. Mirrors the caller's own recreate-on-resize guard timing;
        // it does not decide when to recreate, only how.
        void ensureCullBuffers(
            ID3D12Device5* device,
            uint32_t maxInstances,
            uint32_t maxBins);
        void destroyCullBuffers();
        ID3D12CommandSignature* indirectCommandSignature(
            ID3D12RootSignature* rootSignature);

        // Public: consumed directly by the pass recorders and the graphics
        // draw path, mirroring the style of DX12GraphResourceEntry.
        DX12Buffer visibleInstances;
        DX12Buffer visibleInstanceCount;
        DX12Buffer visibleInstanceCountReadback;
        DX12Buffer indirectArguments;
        DX12Buffer binCounts;
        D3D12_RESOURCE_STATES visibleInstancesState = D3D12_RESOURCE_STATE_COMMON;
        D3D12_RESOURCE_STATES visibleInstanceCountState =
            D3D12_RESOURCE_STATE_COMMON;
        D3D12_RESOURCE_STATES indirectArgumentsState =
            D3D12_RESOURCE_STATE_COMMON;
        D3D12_RESOURCE_STATES binCountsState = D3D12_RESOURCE_STATE_COMMON;
        uint32_t maxInstances = 0;
        uint32_t lastVisibleInstanceCount = 0;
        bool loggedGpuCull = false;

    private:
        DX12ResourceAllocator* m_resourceAllocator = nullptr;
        DX12DescriptorSystem* m_descriptorSystem = nullptr;
        ID3D12Device5* m_device = nullptr;

        std::vector<DX12GpuSceneFrameResources> m_frames;
        PreparedGpuScene m_prepared;
        std::unordered_map<
            ID3D12RootSignature*,
            Microsoft::WRL::ComPtr<ID3D12CommandSignature>>
            m_indirectCommandSignatures;
    };
}
