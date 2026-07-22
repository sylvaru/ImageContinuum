#pragma once

#include <cstdint>
#include <vector>
#include <span>

#include <glm/glm.hpp>

#include "ic/renderer/path_tracing/path_trace_scene_builder.h"

namespace ic
{
    class RayTracingAccelerationStructureProvider;
    struct SceneRenderView;

    // API-neutral capability contract. `sharedAccelerationStructures` is kept
    // separate from the hardware bits: GI must not schedule merely because the
    // adapter advertises ray queries when the renderer has not materialized the
    // shared BLAS/TLAS scene yet.
    struct RayTracingCapabilities
    {
        bool accelerationStructures = false;
        bool inlineRayQueries = false;
        bool accelerationStructureUpdates = false;
        bool indirectDispatch = false;
        bool sharedAccelerationStructures = false;
        // True only when native texture/AS state tracking is proven safe for
        // inline-query passes on the backend's async compute queue.
        bool asyncComputeQueries = false;

        [[nodiscard]] bool supportsInlineSceneQueries() const noexcept
        {
            return accelerationStructures && inlineRayQueries &&
                sharedAccelerationStructures;
        }
    };

    struct RayTracingSceneStatistics
    {
        uint64_t generation = 0;
        uint64_t sceneVersion = UINT64_MAX;
        uint32_t vertexCount = 0;
        uint32_t triangleCount = 0;
        uint32_t materialCount = 0;
        uint32_t bvhNodeCount = 0;
        uint32_t instanceCount = 0;
        uint32_t firstEmissiveInstanceIndex = UINT32_MAX;
        uint32_t rebuildCount = 0;
        bool valid = false;
    };

    enum class RayTracingSceneUpdateKind : uint8_t
    {
        None,
        TlasRefit,
        TlasRebuild,
        GeometryRebuild
    };

    enum RayTracingGeometryFlags : uint32_t
    {
        RayTracingGeometryOpaque = 1u << 0,
        RayTracingGeometryAlphaTested = 1u << 1,
        RayTracingGeometryDoubleSided = 1u << 2,
        RayTracingGeometryProceduralEligible = 1u << 3
    };

    [[nodiscard]] RayTracingSceneUpdateKind classifyRayTracingSceneUpdate(
        bool currentValid,
        uint64_t oldGeometry, uint64_t newGeometry,
        uint64_t oldTopology, uint64_t newTopology,
        uint64_t oldState, uint64_t newState) noexcept;
    [[nodiscard]] uint32_t rayTracingGeometryFlags(
        const MaterialAsset* material) noexcept;

    struct RayTracingGeometryRecord
    {
        AssetHandle model = {};
        uint32_t geometryIndex = UINT32_MAX;
        uint32_t modelBufferIndex = UINT32_MAX;
        uint32_t meshIndex = UINT32_MAX;
        uint32_t primitiveIndex = UINT32_MAX;
        uint32_t firstVertex = 0;
        uint32_t vertexCount = 0;
        uint32_t firstIndex = 0;
        uint32_t indexCount = 0;
        uint32_t materialIndex = 0;
        uint32_t flags = RayTracingGeometryOpaque;
        uint32_t padding0 = 0;
        uint32_t padding1 = 0;
    };

    struct alignas(16) RayTracingInstanceRecord
    {
        glm::mat4 world = glm::mat4(1.0f);
        uint32_t instanceId = UINT32_MAX;
        uint32_t generation = 0;
        uint32_t geometryIndex = UINT32_MAX;
        uint32_t sourceEntity = UINT32_MAX;
        uint32_t mask = 0xffu;
        uint32_t flags = 0;
        uint32_t materialOffset = 0;
        uint32_t padding = 0;
    };
    static_assert(sizeof(RayTracingInstanceRecord) == 96);

    // Stable, API-neutral tables consumed by inline ray-query shaders. The
    // geometry table references the renderer's resident per-model vertex and
    // index buffers; it does not contain a second copy of mesh data.
    struct alignas(16) GpuRayTracingGeometryRecord
    {
        glm::uvec4 vertexIndexRange = {};
        glm::uvec4 mapping = {};
        glm::uvec4 source = {};
    };

    struct alignas(16) GpuRayTracingInstanceRecord
    {
        glm::mat4 objectToWorld = glm::mat4(1.0f);
        glm::mat4 normalToWorld = glm::mat4(1.0f);
        glm::uvec4 identity = {};
        glm::uvec4 state = {};
    };

    static_assert(sizeof(GpuRayTracingGeometryRecord) == 48);
    static_assert(sizeof(GpuRayTracingInstanceRecord) == 160);
    static_assert(sizeof(AssetVertex) == 72);
    static_assert(offsetof(AssetVertex, position) == 0);
    static_assert(offsetof(AssetVertex, normal) == 12);
    static_assert(offsetof(AssetVertex, tangent) == 24);
    static_assert(offsetof(AssetVertex, uv0) == 40);

    // Renderer-owned CPU scene cache used by every ray-using path. Backends may
    // translate this snapshot into native buffers and BLAS/TLAS objects, but
    // they do not rebuild geometry or own a path-specific copy of the scene.
    class RayTracingSceneService final
    {
    public:
        bool update(
            const SceneRenderView& scene,
            const AssetManager& assets,
            const PathTraceMaterialTextureResolver& resolveTexture = {},
            uint64_t materialBindingGeneration = 0);

        // Refreshes the stable native-AS geometry and instance view without
        // rebuilding the path tracer's flattened compute-BVH payload.
        bool updateGeometry(
            const SceneRenderView& scene,
            const AssetManager& assets);

        void adopt(
            uint64_t sceneVersion,
            PathTraceSceneData sceneData,
            uint64_t materialBindingGeneration = 0);

        void invalidate() noexcept;
        void setAccelerationStructureProvider(
            RayTracingAccelerationStructureProvider* provider) noexcept
        {
            m_accelerationStructures = provider;
        }
        [[nodiscard]] RayTracingAccelerationStructureProvider*
            accelerationStructures() const noexcept
        {
            return m_accelerationStructures;
        }

        [[nodiscard]] const PathTraceSceneData& sceneData() const noexcept
        {
            return m_sceneData;
        }

        [[nodiscard]] const RayTracingSceneStatistics& statistics() const noexcept
        {
            return m_statistics;
        }

        [[nodiscard]] const std::vector<RayTracingInstanceRecord>&
            instances() const noexcept
        {
            return m_instances;
        }

        [[nodiscard]] std::span<const RayTracingGeometryRecord>
            geometries() const noexcept
        {
            return m_geometries;
        }

        [[nodiscard]] std::span<const GpuRayTracingGeometryRecord>
            gpuGeometries() const noexcept { return m_gpuGeometries; }
        [[nodiscard]] std::span<const GpuRayTracingInstanceRecord>
            gpuInstances() const noexcept { return m_gpuInstances; }

        [[nodiscard]] RayTracingSceneUpdateKind lastUpdateKind() const noexcept
        {
            return m_lastUpdateKind;
        }

    private:
        PathTraceSceneData m_sceneData;
        std::vector<RayTracingGeometryRecord> m_geometries;
        std::vector<RayTracingInstanceRecord> m_instances;
        std::vector<GpuRayTracingGeometryRecord> m_gpuGeometries;
        std::vector<GpuRayTracingInstanceRecord> m_gpuInstances;
        RayTracingSceneStatistics m_statistics;
        uint64_t m_materialBindingGeneration = UINT64_MAX;
        uint64_t m_geometrySignature = 0;
        uint64_t m_instanceTopologySignature = 0;
        uint64_t m_instanceStateSignature = 0;
        RayTracingSceneUpdateKind m_lastUpdateKind =
            RayTracingSceneUpdateKind::None;
        RayTracingAccelerationStructureProvider* m_accelerationStructures =
            nullptr;
    };
}
