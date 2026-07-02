#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

namespace ic
{
    enum class PathTraceMaterialType : uint32_t
    {
        Diffuse = 0,
        Emissive = 1,
        Metallic = 2,
        Dielectric = 3
    };

    struct PathTracerSettings
    {
        uint32_t samplesPerPixel = 1;
        uint32_t maxBounces = 2;
        float exposure = 1.0f;
        bool accumulate = true;
        bool enableEnvironment = true;
    };

    struct PathTracerRuntimeState
    {
        uint32_t accumulatedSampleCount = 0;
        uint32_t frameIndex = 0;
        bool resetAccumulation = true;

        glm::mat4 previousView{ 1.0f };
        glm::mat4 previousProjection{ 1.0f };

        uint32_t width = 0;
        uint32_t height = 0;
    };

    struct PathTraceCameraData
    {
        glm::mat4 inverseView{ 1.0f };
        glm::mat4 inverseProjection{ 1.0f };
        glm::vec3 cameraPosition{ 0.0f };
        float verticalFov = 0.0f;

        glm::vec2 jitter{ 0.0f };
        glm::vec2 padding{ 0.0f };
    };

    struct PathTraceVertex
    {
        glm::vec4 position{ 0.0f, 0.0f, 0.0f, 1.0f };
        glm::vec4 normal{ 0.0f, 1.0f, 0.0f, 0.0f };
    };

    struct PathTraceMaterial
    {
        glm::vec4 baseColor{ 1.0f };
        glm::vec4 emissive{ 0.0f };
        uint32_t baseColorTextureIndex = UINT32_MAX;
        uint32_t normalTextureIndex = UINT32_MAX;
        PathTraceMaterialType materialType = PathTraceMaterialType::Diffuse;
        float roughness = 1.0f;
        float metallic = 0.0f;
        float padding = 0.0f;
        glm::vec2 padding1{ 0.0f };
    };

    struct PathTraceTriangle
    {
        uint32_t i0 = 0;
        uint32_t i1 = 0;
        uint32_t i2 = 0;
        uint32_t materialIndex = 0;
    };

    struct PathTraceInstance
    {
        glm::mat4 localToWorld{ 1.0f };
        glm::mat4 worldToLocal{ 1.0f };
        uint32_t firstTriangle = 0;
        uint32_t triangleCount = 0;
        uint32_t blasRootNode = UINT32_MAX;
        uint32_t materialOffset = 0;
    };

    struct PathTraceBVHNode
    {
        glm::vec3 boundsMin{ 0.0f };
        uint32_t leftFirst = 0;

        glm::vec3 boundsMax{ 0.0f };
        uint32_t count = 0;
    };

    struct PathTraceSceneData
    {
        std::vector<PathTraceVertex> vertices;
        std::vector<PathTraceMaterial> materials;
        std::vector<PathTraceTriangle> triangles;
        std::vector<PathTraceBVHNode> bvhNodes;
    };

    static_assert(sizeof(PathTraceVertex) == 32);
    static_assert(sizeof(PathTraceMaterial) == 64);
    static_assert(sizeof(PathTraceTriangle) == 16);
    static_assert(sizeof(PathTraceBVHNode) == 32);
}
