#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>

#include "ic/core/asset_manager.h"
#include "ic/renderer/render_handles.h"

namespace ic
{
    struct FrameContext;

    struct GpuFrameData
    {
        glm::mat4 view = glm::mat4(1.0f);
        glm::mat4 projection = glm::mat4(1.0f);
        glm::mat4 viewProjection = glm::mat4(1.0f);

        glm::vec3 cameraPosition = glm::vec3(0.0f);
        float time = 0.0f;

        glm::vec3 lightDirection = glm::vec3(0.0f, -1.0f, 0.0f);
        float lightIntensity = 1.0f;

        glm::vec3 lightColor = glm::vec3(1.0f);
        float padding0 = 0.0f;
    };

    struct GpuObjectData
    {
        glm::mat4 world = glm::mat4(1.0f);
        glm::mat4 inverseTransposeWorld = glm::mat4(1.0f);
    };

    struct GpuMaterialData
    {
        glm::vec4 baseColorFactor = glm::vec4(1.0f);

        float metallicFactor = 1.0f;
        float roughnessFactor = 1.0f;
        float alphaCutoff = 0.5f;
        uint32_t flags = 0;

        uint32_t baseColorTextureIndex = 0;
        uint32_t normalTextureIndex = 2;
        uint32_t metallicRoughnessTextureIndex = 3;
        uint32_t samplerIndex = 0;
    };

    struct GpuMesh
    {
        uint32_t vertexBufferIndex = 0;
        uint32_t indexBufferIndex = 0;
        uint32_t materialIndex = 0;
        uint32_t indexCount = 0;

        uint32_t firstIndex = 0;
        uint32_t vertexOffset = 0;
        uint32_t padding0 = 0;
        uint32_t padding1 = 0;

        uint64_t vertexBufferAddress = 0;
        uint64_t indexBufferAddress = 0;
    };

    struct DrawConstants
    {
        uint32_t objectIndex = 0;
        uint32_t meshIndex = 0;
        uint32_t materialIndex = 0;
        uint32_t flags = 0;
    };

    struct GpuModelHandle
    {
        uint32_t index = UINT32_MAX;
        uint32_t generation = 0;

        explicit operator bool() const
        {
            return index != UINT32_MAX;
        }
    };

    struct GpuModel
    {
        GpuModelHandle handle = {};
        AssetHandle source = {};

        BufferHandle vertexBuffer = {};
        BufferHandle indexBuffer = {};

        std::vector<GpuMesh> meshes;
        std::vector<GpuMaterialData> materials;

        bool uploadQueued = false;
        bool uploaded = false;
    };

    class RendererGpuAssetCache final
    {
    public:
        GpuModelHandle requestModel(AssetHandle model);

        void updateUploads(
            FrameContext& frame,
            AssetManager& assets);

        const GpuModel* model(GpuModelHandle handle) const;

    private:
        std::vector<GpuModel> m_models;
        std::unordered_map<AssetHandle, GpuModelHandle, AssetHandleHash> m_modelLookup;
    };
}
