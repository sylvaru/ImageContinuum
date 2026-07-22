#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>

#include "ic/core/asset_manager.h"
#include "ic/renderer/render_handles.h"
#include "ic/renderer/render_types.h"

namespace ic
{
    inline constexpr uint32_t MaxGpuPointLights = 8;

    struct FrameContext;

    struct GpuOcclusionHistoryState
    {
        glm::mat4 view = glm::mat4(1.0f);
        glm::mat4 viewProjection = glm::mat4(1.0f);
        glm::vec3 cameraPosition = glm::vec3(0.0f);
        float nearPlane = 0.0f;
        float farPlane = 0.0f;
        float verticalFovRadians = 0.0f;
        float aspectRatio = 0.0f;
        uint32_t width = 0;
        uint32_t height = 0;
        uint64_t sceneVersion = UINT64_MAX;
        bool valid = false;
    };

    struct GpuFrameData
    {
        glm::mat4 view = glm::mat4(1.0f);
        glm::mat4 projection = glm::mat4(1.0f);
        glm::mat4 viewProjection = glm::mat4(1.0f);
        glm::mat4 previousView = glm::mat4(1.0f);
        glm::mat4 previousViewProjection = glm::mat4(1.0f);

        glm::vec3 cameraPosition = glm::vec3(0.0f);
        float time = 0.0f;

        glm::vec3 lightDirection = glm::vec3(0.0f, -1.0f, 0.0f);
        float lightIntensity = 1.0f;

        glm::vec3 lightColor = glm::vec3(1.0f);
        float padding0 = 0.0f;

        uint32_t environmentEnabled = 0;
        uint32_t prefilteredMipCount = 1;
        float environmentIntensity = 1.0f;
        float environmentExposure = 1.0f;

        uint32_t pointLightCount = 0;
        float environmentTransportExposure = 1.0f;
        glm::vec2 padding1 = glm::vec2(0.0f);

        glm::vec4 pointLightPositionRange[MaxGpuPointLights] = {};
        glm::vec4 pointLightColorIntensity[MaxGpuPointLights] = {};

        glm::uvec4 clusterDimensions = glm::uvec4(0);
        glm::uvec4 clusterConfig = glm::uvec4(0);
        glm::uvec4 renderExtentAndHiZ = glm::uvec4(0);
        glm::uvec4 cullingConfig = glm::uvec4(0);
        glm::vec4 cameraNearFar = glm::vec4(0.0f);
        glm::vec4 occlusionConfig = glm::vec4(0.0f);
        glm::uvec4 occlusionDebugConfig = glm::uvec4(0);
        // x: resolved diffuse-GI texture valid; y: debug view; z: float bits of
        // diagnostic-only intensity; w reserved.
        glm::uvec4 globalIlluminationConfig = glm::uvec4(0);
    };

    struct GpuObjectData
    {
        glm::mat4 world = glm::mat4(1.0f);
        glm::mat4 inverseTransposeWorld = glm::mat4(1.0f);
    };

    struct GpuMaterialData
    {
        glm::vec4 baseColorFactor = glm::vec4(1.0f);
        glm::vec4 emissiveFactor = glm::vec4(0.0f);

        float metallicFactor = 0.0f;
        float roughnessFactor = 1.0f;
        float alphaCutoff = 0.5f;
        float occlusionStrength = 1.0f;

        uint32_t flags = 0;
        uint32_t baseColorTextureIndex = UINT32_MAX;
        uint32_t normalTextureIndex = UINT32_MAX;
        uint32_t metallicRoughnessTextureIndex = UINT32_MAX;

        uint32_t occlusionTextureIndex = UINT32_MAX;
        uint32_t emissiveTextureIndex = UINT32_MAX;
        uint32_t baseColorSamplerIndex = UINT32_MAX;
        uint32_t normalSamplerIndex = UINT32_MAX;

        uint32_t metallicRoughnessSamplerIndex = UINT32_MAX;
        uint32_t occlusionSamplerIndex = UINT32_MAX;
        uint32_t emissiveSamplerIndex = UINT32_MAX;
        uint32_t padding0 = 0;
    };

    GpuMaterialData makeGpuMaterialData(const MaterialAsset& src);

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

        GpuInstanceBounds bounds = {};
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
