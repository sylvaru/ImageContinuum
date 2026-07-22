#include "ic/common/ic_pch.h"
#include "ic/renderer/ray_tracing/ray_tracing_scene.h"
#include "ic/scene/scene_render_view.h"

#include <glm/gtc/matrix_inverse.hpp>

namespace ic
{
    RayTracingSceneUpdateKind classifyRayTracingSceneUpdate(
        bool currentValid,
        uint64_t oldGeometry, uint64_t newGeometry,
        uint64_t oldTopology, uint64_t newTopology,
        uint64_t oldState, uint64_t newState) noexcept
    {
        if (!currentValid || oldGeometry != newGeometry)
            return RayTracingSceneUpdateKind::GeometryRebuild;
        if (oldTopology != newTopology)
            return RayTracingSceneUpdateKind::TlasRebuild;
        if (oldState != newState)
            return RayTracingSceneUpdateKind::TlasRefit;
        return RayTracingSceneUpdateKind::None;
    }

    uint32_t rayTracingGeometryFlags(const MaterialAsset* material) noexcept
    {
        uint32_t flags = RayTracingGeometryOpaque;
        if (material && (material->alphaMask || material->alphaBlend))
        {
            flags &= ~RayTracingGeometryOpaque;
            flags |= RayTracingGeometryAlphaTested;
        }
        if (material && material->doubleSided)
            flags |= RayTracingGeometryDoubleSided;
        return flags;
    }

    namespace
    {
        void hashCombine(uint64_t& hash, uint64_t value) noexcept
        {
            hash ^= value + 0x9e3779b97f4a7c15ull +
                (hash << 6u) + (hash >> 2u);
        }

        uint64_t hashMatrix(const glm::mat4& matrix) noexcept
        {
            uint64_t hash = 1469598103934665603ull;
            const auto* bytes = reinterpret_cast<const uint8_t*>(&matrix);
            for (size_t i = 0; i < sizeof(matrix); ++i)
            {
                hash ^= bytes[i];
                hash *= 1099511628211ull;
            }
            return hash;
        }
    }

    bool RayTracingSceneService::updateGeometry(
        const SceneRenderView& scene,
        const AssetManager& assets)
    {
        if (m_statistics.valid &&
            m_statistics.sceneVersion == scene.sceneVersion)
        {
            return false;
        }

        std::vector<RayTracingGeometryRecord> geometries;
        std::vector<RayTracingInstanceRecord> instances;
        std::vector<bool> geometryEmissive;
        std::unordered_map<AssetHandle, uint32_t, AssetHandleHash>
            modelFirstGeometry;
        std::unordered_map<AssetHandle, uint32_t, AssetHandleHash>
            modelBufferIndices;
        std::unordered_map<AssetHandle, uint32_t, AssetHandleHash>
            modelMaterialOffsets;
        uint32_t nextModelBufferIndex = 0;
        uint32_t nextMaterialOffset = 0;
        uint64_t geometrySignature = 0;
        uint64_t topologySignature = 0;
        uint64_t stateSignature = 0;

        for (const SceneModelRenderItem& item : scene.models)
        {
            const ModelAsset* model = assets.model(item.model);
            if (!model)
                continue;

            const auto [bufferIt, insertedBuffer] =
                modelBufferIndices.try_emplace(
                    item.model, nextModelBufferIndex);
            if (insertedBuffer)
                ++nextModelBufferIndex;
            const auto [materialIt, insertedMaterial] =
                modelMaterialOffsets.try_emplace(
                    item.model, nextMaterialOffset);
            if (insertedMaterial)
                nextMaterialOffset += static_cast<uint32_t>(
                    std::max<size_t>(model->materials.size(), 1u));
            const uint32_t modelBufferIndex = bufferIt->second;
            const uint32_t materialOffset = materialIt->second;

            uint32_t firstGeometry = 0;
            if (const auto found = modelFirstGeometry.find(item.model);
                found != modelFirstGeometry.end())
            {
                firstGeometry = found->second;
            }
            else
            {
                firstGeometry = static_cast<uint32_t>(geometries.size());
                modelFirstGeometry.emplace(item.model, firstGeometry);
                for (uint32_t meshIndex = 0;
                     meshIndex < model->meshes.size(); ++meshIndex)
                {
                    const MeshAsset& mesh = model->meshes[meshIndex];
                    for (uint32_t primitiveIndex = 0;
                         primitiveIndex < mesh.primitives.size();
                         ++primitiveIndex)
                    {
                        const MeshPrimitiveAsset& primitive =
                            mesh.primitives[primitiveIndex];
                        const uint32_t materialIndex =
                            primitive.materialIndex != UINT32_MAX &&
                            primitive.materialIndex < model->materials.size()
                                ? primitive.materialIndex : 0u;
                        const MaterialAsset* material =
                            materialIndex < model->materials.size()
                                ? &model->materials[materialIndex] : nullptr;
                        const uint32_t flags =
                            rayTracingGeometryFlags(material);
                        const bool emissive = material &&
                            glm::dot(material->emissiveFactor *
                                material->emissiveStrength,
                                material->emissiveFactor *
                                material->emissiveStrength) > 0.0f;
                        RayTracingGeometryRecord geometry{
                            .model = item.model,
                            .geometryIndex =
                                static_cast<uint32_t>(geometries.size()),
                            .modelBufferIndex = modelBufferIndex,
                            .meshIndex = meshIndex,
                            .primitiveIndex = primitiveIndex,
                            .firstVertex = primitive.firstVertex,
                            .vertexCount = primitive.vertexCount,
                            .firstIndex = primitive.firstIndex,
                            .indexCount = primitive.indexCount,
                            .materialIndex = materialIndex,
                            .flags = flags };
                        geometries.push_back(geometry);
                        geometryEmissive.push_back(emissive);
                        hashCombine(geometrySignature, item.model.index);
                        hashCombine(geometrySignature, item.model.generation);
                        hashCombine(geometrySignature, geometry.modelBufferIndex);
                        hashCombine(geometrySignature, geometry.firstVertex);
                        hashCombine(geometrySignature, geometry.vertexCount);
                        hashCombine(geometrySignature, geometry.firstIndex);
                        hashCombine(geometrySignature, geometry.indexCount);
                        hashCombine(geometrySignature, geometry.materialIndex);
                        hashCombine(geometrySignature, geometry.flags);
                    }
                }
            }

            std::vector<glm::mat4> meshTransforms(
                model->meshes.size(), glm::mat4(1.0f));
            for (const NodeAsset& node : model->nodes)
            {
                if (node.meshIndex >= 0 &&
                    static_cast<size_t>(node.meshIndex) < meshTransforms.size())
                {
                    meshTransforms[static_cast<size_t>(node.meshIndex)] =
                        node.worldMatrix;
                }
            }

            uint32_t modelGeometryOffset = firstGeometry;
            for (uint32_t meshIndex = 0;
                 meshIndex < model->meshes.size(); ++meshIndex)
            {
                const MeshAsset& mesh = model->meshes[meshIndex];
                for (uint32_t primitiveIndex = 0;
                     primitiveIndex < mesh.primitives.size();
                     ++primitiveIndex, ++modelGeometryOffset)
                {
                    const uint32_t instanceId =
                        static_cast<uint32_t>(instances.size());
                    const glm::mat4 world =
                        item.world * meshTransforms[meshIndex];
                    RayTracingInstanceRecord instance{
                        .world = world,
                        .instanceId = instanceId,
                        .generation = static_cast<uint32_t>(scene.sceneVersion) ^
                            item.model.generation,
                        .geometryIndex = modelGeometryOffset,
                        .sourceEntity = static_cast<uint32_t>(item.entity),
                        .mask = 0xffu,
                        .flags = item.flags,
                        .materialOffset = materialOffset };
                    instances.push_back(instance);
                    hashCombine(topologySignature, instance.sourceEntity);
                    hashCombine(topologySignature, instance.geometryIndex);
                    hashCombine(topologySignature, instance.mask);
                    hashCombine(stateSignature, hashMatrix(instance.world));
                    hashCombine(stateSignature, instance.generation);
                    hashCombine(stateSignature, instance.flags);
                }
            }
        }

        m_lastUpdateKind = classifyRayTracingSceneUpdate(
            m_statistics.valid,
            m_geometrySignature, geometrySignature,
            m_instanceTopologySignature, topologySignature,
            m_instanceStateSignature, stateSignature);

        m_geometrySignature = geometrySignature;
        m_instanceTopologySignature = topologySignature;
        m_instanceStateSignature = stateSignature;
        m_geometries = std::move(geometries);
        m_instances = std::move(instances);
        m_gpuGeometries.clear();
        m_gpuGeometries.reserve(m_geometries.size());
        for (const RayTracingGeometryRecord& geometry : m_geometries)
        {
            m_gpuGeometries.push_back({
                .vertexIndexRange = glm::uvec4(
                    geometry.firstVertex, geometry.vertexCount,
                    geometry.firstIndex, geometry.indexCount),
                .mapping = glm::uvec4(
                    geometry.modelBufferIndex, geometry.materialIndex,
                    geometry.flags, geometry.geometryIndex),
                .source = glm::uvec4(
                    geometry.model.index, geometry.model.generation,
                    geometry.meshIndex, geometry.primitiveIndex) });
        }
        m_gpuInstances.clear();
        m_gpuInstances.reserve(m_instances.size());
        m_statistics.firstEmissiveInstanceIndex = UINT32_MAX;
        for (const RayTracingInstanceRecord& instance : m_instances)
        {
            if (m_statistics.firstEmissiveInstanceIndex == UINT32_MAX &&
                instance.geometryIndex < geometryEmissive.size() &&
                geometryEmissive[instance.geometryIndex])
            {
                m_statistics.firstEmissiveInstanceIndex = instance.instanceId;
            }
            m_gpuInstances.push_back({
                .objectToWorld = instance.world,
                .normalToWorld = glm::inverseTranspose(instance.world),
                .identity = glm::uvec4(
                    instance.instanceId, instance.generation,
                    instance.geometryIndex, instance.sourceEntity),
                .state = glm::uvec4(
                    instance.mask, instance.flags,
                    instance.materialOffset, 0u) });
        }
        ++m_statistics.generation;
        ++m_statistics.rebuildCount;
        m_statistics.sceneVersion = scene.sceneVersion;
        m_statistics.instanceCount =
            static_cast<uint32_t>(m_instances.size());
        m_statistics.valid = !m_geometries.empty() && !m_instances.empty();
        return true;
    }

    bool RayTracingSceneService::update(
        const SceneRenderView& scene,
        const AssetManager& assets,
        const PathTraceMaterialTextureResolver& resolveTexture,
        uint64_t materialBindingGeneration)
    {
        const bool geometryChanged = updateGeometry(scene, assets);
        if (!geometryChanged &&
            m_materialBindingGeneration == materialBindingGeneration &&
            !m_sceneData.bvhNodes.empty())
        {
            return false;
        }

        m_sceneData = buildPathTraceSceneData(scene, assets, resolveTexture);
        m_materialBindingGeneration = materialBindingGeneration;
        m_statistics.vertexCount =
            static_cast<uint32_t>(m_sceneData.vertices.size());
        m_statistics.triangleCount =
            static_cast<uint32_t>(m_sceneData.triangles.size());
        m_statistics.materialCount =
            static_cast<uint32_t>(m_sceneData.materials.size());
        m_statistics.bvhNodeCount =
            static_cast<uint32_t>(m_sceneData.bvhNodes.size());
        return true;
    }

    void RayTracingSceneService::invalidate() noexcept
    {
        m_statistics.valid = false;
        m_statistics.sceneVersion = UINT64_MAX;
        m_materialBindingGeneration = UINT64_MAX;
        m_lastUpdateKind = RayTracingSceneUpdateKind::GeometryRebuild;
        m_statistics.firstEmissiveInstanceIndex = UINT32_MAX;
    }

    void RayTracingSceneService::adopt(
        uint64_t sceneVersion,
        PathTraceSceneData sceneData,
        uint64_t materialBindingGeneration)
    {
        m_sceneData = std::move(sceneData);
        m_instances.clear();
        m_geometries.clear();
        m_gpuInstances.clear();
        m_gpuGeometries.clear();
        m_materialBindingGeneration = materialBindingGeneration;
        ++m_statistics.generation;
        ++m_statistics.rebuildCount;
        m_statistics.sceneVersion = sceneVersion;
        m_statistics.vertexCount =
            static_cast<uint32_t>(m_sceneData.vertices.size());
        m_statistics.triangleCount =
            static_cast<uint32_t>(m_sceneData.triangles.size());
        m_statistics.materialCount =
            static_cast<uint32_t>(m_sceneData.materials.size());
        m_statistics.bvhNodeCount =
            static_cast<uint32_t>(m_sceneData.bvhNodes.size());
        m_statistics.firstEmissiveInstanceIndex = UINT32_MAX;
        m_statistics.instanceCount =
            static_cast<uint32_t>(m_instances.size());
        m_statistics.valid = !m_sceneData.vertices.empty() &&
            !m_sceneData.triangles.empty() && !m_sceneData.bvhNodes.empty();
    }
}
