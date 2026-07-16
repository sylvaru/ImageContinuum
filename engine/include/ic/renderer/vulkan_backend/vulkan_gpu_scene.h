#pragma once

#include "ic/renderer/vulkan_backend/vulkan_resource_allocator.h"
#include "ic/renderer/renderer_gpu_assets.h"
#include "ic/renderer/render_pipeline.h"
#include "ic/renderer/gpu_scene_preparation.h"
#include "ic/core/asset_manager.h"
#include "ic/scene/scene_render_view.h"

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include <vulkan/vulkan.h>

namespace ic
{
    // GPU-resident representation of an uploaded model. Shared vocabulary
    // between the backend's asset-upload cache (which owns residency) and the
    // GPU scene's per-frame draw preparation (which only references it).
    struct VulkanUploadedModel
    {
        VulkanBuffer vertexBuffer;
        VulkanBuffer indexBuffer;
        std::vector<GpuMesh> meshes;
        std::vector<glm::mat4> meshTransforms;
        std::vector<GpuMaterialData> materials;
        std::vector<uint32_t> textureDescriptorIndices;
        std::vector<uint32_t> samplerDescriptorIndices;
        bool uploaded = false;
    };

    // Persistent, growth-only per-frame-slot scene upload buffers and the
    // combined scene descriptor set consumed by the forward/clustered-forward
    // pipelines. The descriptor set itself is (re)built by the backend
    // (updateFrameDescriptors) since it spans bindless texture/sampler
    // residency and environment state that GpuScene does not own.
    struct VulkanGpuSceneFrameResources
    {
        VulkanBuffer frameConstants;
        VulkanBuffer objects;
        VulkanBuffer materials;
        // visibleLights and the GPU-driven cull inputs are NOT here:
        // they are frame-graph-registry owned (per frame slot) and uploaded into
        // by the backend, so there is no duplicate backend allocation.

        VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
        VkDescriptorPool hiZDescriptorPool = VK_NULL_HANDLE;
        VkDescriptorPool gpuCullDescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

        uint32_t objectCapacity = 0;
        uint32_t materialCapacity = 0;
        uint32_t bindlessTextureCount = 0;
        uint32_t bindlessSamplerCount = 0;
        uint64_t environmentVersion = UINT64_MAX;
        bool iblBaked = false;
        PipelineBindingLayoutKind descriptorLayout =
            PipelineBindingLayoutKind::Unknown;
    };

    // Owns the persistent GPU-driven scene: CPU-side draw extraction/sorting/
    // binning, the growth-only per-frame upload buffers, and the GPU-driven
    // cull/indirect-argument buffers consumed by the frustum-cull compute pass
    // and (once activated) indirect draw consumption. The frustum-cull compute
    // shader writes native indirect-argument records directly, so metadata
    // draw commands never cross into the native indirect API.
    class VulkanGpuScene final
    {
    public:
        using DrawItem = PreparedGpuDraw;
        using GeometryBin = PreparedGpuGeometryBin;

        struct PrepareResult
        {
            bool hasData = false;
            // True when the objects/materials/visibleLights buffers were
            // (re)created this call, so the caller's combined descriptor set
            // needs rebuilding. Mirrors the pre-decomposition `descriptorsDirty`
            // flag; deliberately does not cover instanceBounds/drawInputs
            // growth, matching prior behavior.
            bool descriptorsDirty = false;
        };

        // Maps a scene item's AssetHandle to its uploaded GPU model view for
        // CPU-side extraction (spans borrowed from the caller's residency
        // storage). Residency is owned by the caller, not the GpuScene.
        using ResolveModelFn = std::function<GpuSceneModelView(AssetHandle)>;
        using BuildFrameDataFn = std::function<GpuFrameData(
            uint32_t visibleLightCount,
            uint32_t instanceBoundsCount,
            uint32_t geometryBinCount)>;

        VulkanGpuScene() = default;
        VulkanGpuScene(const VulkanGpuScene&) = delete;
        VulkanGpuScene& operator=(const VulkanGpuScene&) = delete;

        void init(
            VulkanResourceAllocator& resourceAllocator,
            uint32_t framesInFlight);
        void shutdown(VkDevice device);

        // Extracts, sorts, bins and uploads this frame's scene draw data.
        // Cached per frameIndex: a second call for the same frame only
        // re-evaluates the caller's descriptor-set dirtiness. `resolveModel`
        // maps a scene item's AssetHandle to its uploaded GPU model (residency
        // is owned by the caller). `buildFrameData` is invoked once extraction
        // is complete (so it can read final instance/geometry-bin counts).
        PrepareResult prepare(
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
        // Prepared GPU-driven cull inputs, uploaded by the backend into the
        // graph-owned per-frame-slot buffers.
        [[nodiscard]] std::span<const GpuInstanceBounds>
            preparedInstanceBounds() const noexcept
        {
            return m_prepared.instanceBounds();
        }
        [[nodiscard]] std::span<const GpuDrawInput>
            preparedDrawInputs() const noexcept
        {
            return m_prepared.drawInputs();
        }
        [[nodiscard]] std::span<const GpuVisibleLight>
            preparedVisibleLights() const noexcept
        {
            return m_prepared.visibleLights();
        }

        [[nodiscard]] VulkanGpuSceneFrameResources& frameResources(
            uint32_t frameSlot) noexcept
        {
            return m_frames[frameSlot];
        }
        [[nodiscard]] const VulkanGpuSceneFrameResources& frameResources(
            uint32_t frameSlot) const noexcept
        {
            return m_frames[frameSlot];
        }
        [[nodiscard]] uint32_t frameSlotCount() const noexcept
        {
            return static_cast<uint32_t>(m_frames.size());
        }

        // The GPU-driven cull/indirect buffers are owned by the frame-graph
        // registry, not this class.
        bool loggedGpuCull = false;

    private:
        VulkanResourceAllocator* m_resourceAllocator = nullptr;

        std::vector<VulkanGpuSceneFrameResources> m_frames;
        PreparedGpuScene m_prepared;
    };
}
