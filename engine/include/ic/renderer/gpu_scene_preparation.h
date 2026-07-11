#pragma once

#include "ic/core/asset_manager.h"
#include "ic/renderer/renderer_gpu_assets.h"
#include "ic/scene/scene_render_view.h"

#include <algorithm>
#include <span>
#include <unordered_map>
#include <vector>

#include <glm/gtc/matrix_inverse.hpp>

namespace ic
{
    struct GpuSceneModelView
    {
        AssetHandle source = {};
        std::span<const GpuMesh> meshes;
        std::span<const glm::mat4> meshTransforms;
        std::span<const GpuMaterialData> materials;
    };

    struct PreparedGpuDraw
    {
        AssetHandle model = {};
        uint32_t objectIndex = 0;
        uint32_t meshIndex = 0;
        uint32_t materialIndex = 0;
    };

    struct PreparedGpuGeometryBin
    {
        AssetHandle model = {};
        uint32_t commandOffset = 0;
        uint32_t maxDrawCount = 0;
    };

    // API-neutral CPU result consumed by both native GPU-scene uploaders.
    // Storage is retained across frames so rebuilding the view does not add
    // per-frame allocations after capacities stabilize.
    class PreparedGpuScene final
    {
    public:
        template<typename ResolveModel>
        bool build(
            uint64_t frameIndex,
            const SceneRenderView& scene,
            ResolveModel&& resolveModel)
        {
            if (m_frameIndex == frameIndex && m_valid)
            {
                return true;
            }

            m_frameIndex = frameIndex;
            m_valid = false;
            m_draws.clear();
            m_objects.clear();
            m_instanceBounds.clear();
            m_drawInputs.clear();
            m_geometryBins.clear();
            m_materials.clear();
            m_visibleLights.clear();
            m_materialOffsets.clear();
            m_modelViews.clear();

            if (scene.camera.valid == 0)
            {
                return false;
            }

            m_objects.reserve(scene.models.size());
            m_instanceBounds.reserve(scene.models.size());
            m_materialOffsets.reserve(scene.models.size());
            m_modelViews.reserve(scene.models.size());

            for (const SceneModelRenderItem& item : scene.models)
            {
                const GpuSceneModelView model = resolveModel(item.model);
                if (!model.source || model.meshes.empty())
                {
                    continue;
                }

                // Spans remain valid because native uploaded-model storage is
                // resolved serially and is not mutated during recording.
                m_modelViews.insert_or_assign(model.source, model);

                uint32_t materialOffset = 0;
                if (const auto found = m_materialOffsets.find(model.source);
                    found != m_materialOffsets.end())
                {
                    materialOffset = found->second;
                }
                else
                {
                    materialOffset = static_cast<uint32_t>(m_materials.size());
                    m_materialOffsets.emplace(model.source, materialOffset);
                    m_materials.insert(
                        m_materials.end(),
                        model.materials.begin(),
                        model.materials.end());
                }

                for (uint32_t meshIndex = 0;
                     meshIndex < model.meshes.size();
                     ++meshIndex)
                {
                    const GpuMesh& mesh = model.meshes[meshIndex];
                    const glm::mat4 meshWorld =
                        meshIndex < model.meshTransforms.size()
                            ? item.world * model.meshTransforms[meshIndex]
                            : item.world;
                    const uint32_t objectIndex =
                        static_cast<uint32_t>(m_objects.size());

                    GpuObjectData object{};
                    object.world = meshWorld;
                    object.inverseTransposeWorld =
                        glm::inverseTranspose(meshWorld);
                    m_objects.push_back(object);

                    const glm::vec4 localSphere = mesh.bounds.centerRadius;
                    const glm::vec3 worldCenter = glm::vec3(
                        meshWorld * glm::vec4(glm::vec3(localSphere), 1.0f));
                    const float maxScale = std::max(
                        glm::length(glm::vec3(meshWorld[0])),
                        std::max(
                            glm::length(glm::vec3(meshWorld[1])),
                            glm::length(glm::vec3(meshWorld[2]))));
                    m_instanceBounds.push_back({
                        .centerRadius = glm::vec4(
                            worldCenter, localSphere.w * maxScale) });

                    m_draws.push_back({
                        .model = model.source,
                        .objectIndex = objectIndex,
                        .meshIndex = meshIndex,
                        .materialIndex = materialOffset + mesh.materialIndex });
                }
            }

            if (m_draws.empty())
            {
                return false;
            }

            std::stable_sort(
                m_draws.begin(),
                m_draws.end(),
                [](const PreparedGpuDraw& lhs, const PreparedGpuDraw& rhs)
                {
                    if (lhs.model.index != rhs.model.index)
                    {
                        return lhs.model.index < rhs.model.index;
                    }
                    if (lhs.model.generation != rhs.model.generation)
                    {
                        return lhs.model.generation < rhs.model.generation;
                    }
                    if (lhs.materialIndex != rhs.materialIndex)
                    {
                        return lhs.materialIndex < rhs.materialIndex;
                    }
                    return lhs.meshIndex < rhs.meshIndex;
                });

            m_sortedBounds.clear();
            m_sortedBounds.reserve(m_draws.size());
            m_drawInputs.reserve(m_draws.size());
            AssetHandle currentModel = {};
            uint32_t currentBinIndex = UINT32_MAX;
            for (const PreparedGpuDraw& draw : m_draws)
            {
                if (draw.model != currentModel)
                {
                    currentModel = draw.model;
                    currentBinIndex =
                        static_cast<uint32_t>(m_geometryBins.size());
                    m_geometryBins.push_back({
                        .model = draw.model,
                        .commandOffset =
                            static_cast<uint32_t>(m_drawInputs.size()),
                        .maxDrawCount = 0 });
                }

                const auto viewIt = m_modelViews.find(draw.model);
                if (viewIt == m_modelViews.end() ||
                    draw.meshIndex >= viewIt->second.meshes.size())
                {
                    continue;
                }

                PreparedGpuGeometryBin& bin =
                    m_geometryBins[currentBinIndex];
                const GpuMesh& mesh = viewIt->second.meshes[draw.meshIndex];
                GpuDrawInput input{};
                input.metadata.meshIndex = draw.meshIndex;
                input.metadata.materialIndex = draw.materialIndex;
                input.metadata.transformIndex = draw.objectIndex;
                input.metadata.instanceIndex = draw.objectIndex;
                input.metadata.geometryRangeIndex = draw.meshIndex;
                input.metadata.materialBinIndex = draw.materialIndex;
                input.metadata.geometryBinIndex = currentBinIndex;
                input.indexCount = mesh.indexCount;
                input.firstIndex = mesh.firstIndex;
                input.vertexOffset = 0;
                input.commandBinOffset = bin.commandOffset;
                m_drawInputs.push_back(input);
                m_sortedBounds.push_back(m_instanceBounds[draw.objectIndex]);
                ++bin.maxDrawCount;
            }
            m_instanceBounds.swap(m_sortedBounds);

            if (m_materials.empty())
            {
                m_materials.push_back(GpuMaterialData{});
            }

            m_visibleLights.reserve(std::min<size_t>(
                scene.lights.size(), ClusteredForwardMaxVisibleLights));
            for (const SceneLightRenderItem& light : scene.lights)
            {
                if (light.type != LightType::Point ||
                    m_visibleLights.size() >= ClusteredForwardMaxVisibleLights)
                {
                    continue;
                }
                m_visibleLights.push_back({
                    .positionRange = glm::vec4(light.position, light.range),
                    .colorIntensity = glm::vec4(light.color, light.intensity) });
            }
            if (m_visibleLights.empty())
            {
                m_visibleLights.push_back(GpuVisibleLight{});
            }

            m_valid = true;
            return true;
        }

        uint64_t frameIndex() const { return m_frameIndex; }
        bool valid() const { return m_valid; }
        std::span<const PreparedGpuDraw> draws() const { return m_draws; }
        std::span<const GpuObjectData> objects() const { return m_objects; }
        std::span<const GpuInstanceBounds> instanceBounds() const
        {
            return m_instanceBounds;
        }
        std::span<const GpuDrawInput> drawInputs() const
        {
            return m_drawInputs;
        }
        std::span<const PreparedGpuGeometryBin> geometryBins() const
        {
            return m_geometryBins;
        }
        std::span<const GpuMaterialData> materials() const
        {
            return m_materials;
        }
        std::span<const GpuVisibleLight> visibleLights() const
        {
            return m_visibleLights;
        }

    private:
        uint64_t m_frameIndex = UINT64_MAX;
        bool m_valid = false;
        std::vector<PreparedGpuDraw> m_draws;
        std::vector<GpuObjectData> m_objects;
        std::vector<GpuInstanceBounds> m_instanceBounds;
        std::vector<GpuInstanceBounds> m_sortedBounds;
        std::vector<GpuDrawInput> m_drawInputs;
        std::vector<PreparedGpuGeometryBin> m_geometryBins;
        std::vector<GpuMaterialData> m_materials;
        std::vector<GpuVisibleLight> m_visibleLights;
        std::unordered_map<AssetHandle, uint32_t, AssetHandleHash>
            m_materialOffsets;
        std::unordered_map<AssetHandle, GpuSceneModelView, AssetHandleHash>
            m_modelViews;
    };
}
