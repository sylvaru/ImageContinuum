#pragma once
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "ic/core/asset_manager.h"
#include "ic/renderer/frame_graph/compiled_graph_plan.h"
#include "ic/renderer/frame_graph/frame_graph_pass.h"
#include "ic/scene/scene_render_view.h"

namespace ic
{
    inline constexpr uint32_t DefaultPathTraceMaxBounces = 4;
    inline constexpr uint32_t DefaultPathTraceSamplesPerPixel = 2;
    inline constexpr uint32_t MaxPathTracePointLights = 8;

    struct PathTraceConstants
    {
        uint32_t renderWidth = 0;
        uint32_t renderHeight = 0;
        uint32_t frameIndex = 0;
        uint32_t accumulatedSampleCount = 0;

        float exposure = 1.0f;
        uint32_t resetAccumulation = 1;
        uint32_t maxBounces = 4;
        uint32_t samplesPerPixel = 1;

        uint32_t sceneVertexCount = 0;
        uint32_t sceneMaterialCount = 0;
        uint32_t sceneTriangleCount = 0;
        uint32_t sceneBvhNodeCount = 0;

        uint32_t useSceneGeometry = 0;
        uint32_t environmentEnabled = 0;
        uint32_t sceneEmissiveTriangleIndex = UINT32_MAX;
        uint32_t paddingScene0 = 0;

        float environmentIntensity = 1.0f;
        float environmentExposure = 1.0f;
        glm::vec2 paddingScene1 = glm::vec2(0.0f);

        glm::vec4 cameraPositionAndTanHalfFov = glm::vec4(0.0f);
        glm::vec4 cameraForwardAndAspect = glm::vec4(0.0f);
        glm::vec4 cameraRightAndNear = glm::vec4(0.0f);
        glm::vec4 cameraUpAndFar = glm::vec4(0.0f);

        uint32_t pointLightCount = 0;
        glm::vec3 padding0 = glm::vec3(0.0f);

        glm::vec4 pointLightPositionRange[MaxPathTracePointLights] = {};
        glm::vec4 pointLightColorIntensity[MaxPathTracePointLights] = {};
    };

    struct TonemapConstants
    {
        uint32_t renderWidth = 0;
        uint32_t renderHeight = 0;
        float exposure = 1.0f;
        uint32_t padding0 = 0;
    };

    struct SkyboxConstants
    {
        glm::vec4 cameraPositionAndTanHalfFov = glm::vec4(0.0f);
        glm::vec4 cameraForwardAndAspect = glm::vec4(0.0f);
        glm::vec4 cameraRightAndNear = glm::vec4(0.0f);
        glm::vec4 cameraUpAndFar = glm::vec4(0.0f);
        float intensity = 1.0f;
        float exposure = 1.0f;
        glm::vec2 padding0 = glm::vec2(0.0f);
    };

    static_assert(sizeof(PathTraceConstants) == 416);
    static_assert(sizeof(TonemapConstants) == 16);
    static_assert(sizeof(SkyboxConstants) == 80);

    inline uint64_t alignConstantBufferSize(
        uint64_t size,
        uint64_t alignment = 256)
    {
        return (size + alignment - 1u) & ~(alignment - 1u);
    }

    inline uint64_t uploadedTextureKey(
        AssetHandle modelHandle,
        uint32_t imageIndex,
        TextureTransferFunction transfer)
    {
        return (static_cast<uint64_t>(modelHandle.index) << 33ull) |
            (static_cast<uint64_t>(imageIndex) << 1ull) |
            (transfer == TextureTransferFunction::SRGB ? 1ull : 0ull);
    }

    inline uint64_t samplerKey(const SamplerAsset* sampler)
    {
        if (!sampler)
        {
            return 0;
        }

        uint64_t key = 1469598103934665603ull;
        auto mix = [&](uint64_t value)
        {
            key ^= value;
            key *= 1099511628211ull;
        };
        mix(static_cast<uint32_t>(sampler->minFilter));
        mix(static_cast<uint32_t>(sampler->magFilter));
        mix(static_cast<uint32_t>(sampler->wrapU));
        mix(static_cast<uint32_t>(sampler->wrapV));
        return key;
    }

    inline uint64_t imageByteSize(const ImageAsset& image)
    {
        const uint64_t componentBytes =
            image.format == ImageFormat::RGBA32F ? sizeof(float) : 1u;
        return static_cast<uint64_t>(image.width) *
            static_cast<uint64_t>(image.height) *
            static_cast<uint64_t>(image.channels) *
            componentBytes;
    }

    inline bool matricesDiffer(
        const glm::mat4& a,
        const glm::mat4& b,
        float epsilon = 1.0e-5f)
    {
        for (glm::length_t column = 0; column < 4; ++column)
        {
            for (glm::length_t row = 0; row < 4; ++row)
            {
                if (std::fabs(a[column][row] - b[column][row]) > epsilon)
                {
                    return true;
                }
            }
        }

        return false;
    }

    inline void fillPathTraceCameraConstants(
        const SceneCameraView& camera,
        uint32_t width,
        uint32_t height,
        PathTraceConstants& constants)
    {
        const float fallbackAspect =
            height == 0u
                ? 1.0f
                : static_cast<float>(width) / static_cast<float>(height);

        if (camera.valid != 0u)
        {
            const glm::mat4 inverseView = glm::inverse(camera.view);
            const glm::vec3 right =
                glm::normalize(glm::vec3(inverseView[0]));
            const glm::vec3 up =
                glm::normalize(glm::vec3(inverseView[1]));
            const glm::vec3 forward =
                glm::normalize(-glm::vec3(inverseView[2]));
            const float aspect =
                camera.aspectRatio > 0.0f
                    ? camera.aspectRatio
                    : fallbackAspect;
            const float tanHalfFov =
                std::tan(camera.verticalFovRadians * 0.5f);

            constants.cameraPositionAndTanHalfFov =
                glm::vec4(camera.position, tanHalfFov);
            constants.cameraForwardAndAspect =
                glm::vec4(forward, aspect);
            constants.cameraRightAndNear =
                glm::vec4(right, camera.nearPlane);
            constants.cameraUpAndFar =
                glm::vec4(up, camera.farPlane);
            return;
        }

        const glm::vec3 origin(0.0f, 1.05f, -3.25f);
        const glm::vec3 target(0.0f, 0.95f, 0.95f);
        const glm::vec3 forward = glm::normalize(target - origin);
        const glm::vec3 right =
            glm::normalize(
                glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
        const glm::vec3 up = glm::normalize(glm::cross(right, forward));

        constants.cameraPositionAndTanHalfFov =
            glm::vec4(origin, std::tan(glm::radians(39.0f) * 0.5f));
        constants.cameraForwardAndAspect =
            glm::vec4(forward, fallbackAspect);
        constants.cameraRightAndNear =
            glm::vec4(right, 0.1f);
        constants.cameraUpAndFar =
            glm::vec4(up, 100.0f);
    }

    inline void fillSkyboxConstants(
        const SceneCameraView& camera,
        uint32_t width,
        uint32_t height,
        const EnvironmentSettings& environment,
        SkyboxConstants& constants)
    {
        PathTraceConstants pathConstants{};
        fillPathTraceCameraConstants(camera, width, height, pathConstants);
        constants.cameraPositionAndTanHalfFov =
            pathConstants.cameraPositionAndTanHalfFov;
        constants.cameraForwardAndAspect =
            pathConstants.cameraForwardAndAspect;
        constants.cameraRightAndNear =
            pathConstants.cameraRightAndNear;
        constants.cameraUpAndFar =
            pathConstants.cameraUpAndFar;
        constants.intensity = environment.intensity;
        constants.exposure = environment.skyboxExposure;
    }

    inline bool planUsesPathTracing(const CompiledGraphPlan& plan)
    {
        for (const auto& payload : plan.payloads)
        {
            if (std::get_if<PathTracePassData>(&payload))
            {
                return true;
            }
        }

        return false;
    }


}
