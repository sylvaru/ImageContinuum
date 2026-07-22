#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <cstdlib>
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
#include "ic/renderer/renderer_gpu_assets.h"
#include "ic/scene/scene_render_view.h"

namespace ic
{
    inline constexpr uint32_t DefaultPathTraceMaxBounces = 4;
    inline constexpr uint32_t DefaultPathTraceSamplesPerPixel = 2;
    inline constexpr uint32_t MaxPathTracePointLights = 8;

    inline uint32_t configuredPathTraceMaxBounces() noexcept
    {
        static const uint32_t value = []
        {
            const char* text = nullptr;
#if defined(_MSC_VER)
            char* ownedText = nullptr;
            size_t textLength = 0;
            if (_dupenv_s(&ownedText, &textLength,
                    "IC_PATH_TRACE_MAX_BOUNCES") != 0)
                return DefaultPathTraceMaxBounces;
            text = ownedText;
#else
            text = std::getenv("IC_PATH_TRACE_MAX_BOUNCES");
#endif
            uint32_t result = DefaultPathTraceMaxBounces;
            if (!text || *text == '\0')
            {
#if defined(_MSC_VER)
                std::free(ownedText);
#endif
                return result;
            }
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(text, &end, 10);
            if (end != text && *end == '\0')
                result = std::clamp(static_cast<uint32_t>(parsed), 1u, 8u);
#if defined(_MSC_VER)
            std::free(ownedText);
#endif
            return result;
        }();
        return value;
    }

    inline uint32_t configuredPathTraceReferenceMode() noexcept
    {
        static const uint32_t value = []
        {
            const char* text = nullptr;
#if defined(_MSC_VER)
            char* ownedText = nullptr;
            size_t textLength = 0;
            if (_dupenv_s(&ownedText, &textLength,
                    "IC_PATH_TRACE_INDIRECT_ONLY") != 0)
                return 0u;
            text = ownedText;
#else
            text = std::getenv("IC_PATH_TRACE_INDIRECT_ONLY");
#endif
            const uint32_t result =
                text && text[0] == '1' && text[1] == '\0' ? 1u : 0u;
#if defined(_MSC_VER)
            std::free(ownedText);
#endif
            return result;
        }();
        return value;
    }

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
        uint32_t referenceMode = 0;

        float environmentIntensity = 1.0f;
        float environmentExposure = 1.0f;
        uint32_t sceneEmissiveTriangleCount = 0;
        uint32_t paddingScene1 = 0;

        glm::vec4 cameraPositionAndTanHalfFov = glm::vec4(0.0f);
        glm::vec4 cameraForwardAndAspect = glm::vec4(0.0f);
        glm::vec4 cameraRightAndNear = glm::vec4(0.0f);
        glm::vec4 cameraUpAndFar = glm::vec4(0.0f);

        uint32_t pointLightCount = 0;
        glm::vec3 padding0 = glm::vec3(0.0f);

        glm::vec4 pointLightPositionRange[MaxPathTracePointLights] = {};
        glm::vec4 pointLightColorIntensity[MaxPathTracePointLights] = {};

        glm::vec4 directionalLightDirectionIntensity = glm::vec4(0.0f);
        glm::vec4 directionalLightColor = glm::vec4(0.0f);
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

    static_assert(sizeof(PathTraceConstants) == 448);
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

    struct ImageMipLevel
    {
        uint32_t width = 1;
        uint32_t height = 1;
        uint64_t offset = 0;
        uint64_t size = 0;
    };

    struct ImageMipChain
    {
        std::vector<std::byte> pixels;
        std::vector<ImageMipLevel> levels;
    };

    inline uint32_t completeMipCount(uint32_t width, uint32_t height)
    {
        uint32_t levels = 1;
        for (uint32_t extent = std::max(width, height); extent > 1u;
             extent >>= 1u)
            ++levels;
        return levels;
    }

    // Complete, backend-independent mip generation for model textures. sRGB
    // colors are filtered in linear space; data and normal maps remain linear.
    inline ImageMipChain buildImageMipChain(const ImageAsset& image)
    {
        ImageMipChain result{};
        if (!image.valid())
            return result;

        const uint32_t componentBytes =
            image.format == ImageFormat::RGBA32F ? sizeof(float) : 1u;
        const uint32_t pixelBytes = image.channels * componentBytes;
        const uint32_t mipCount = completeMipCount(image.width, image.height);
        result.levels.reserve(mipCount);

        uint64_t totalBytes = 0;
        uint32_t width = image.width;
        uint32_t height = image.height;
        for (uint32_t mip = 0; mip < mipCount; ++mip)
        {
            const uint64_t levelBytes = static_cast<uint64_t>(width) * height *
                pixelBytes;
            result.levels.push_back({ width, height, totalBytes, levelBytes });
            totalBytes += levelBytes;
            width = std::max(1u, width >> 1u);
            height = std::max(1u, height >> 1u);
        }
        result.pixels.resize(static_cast<size_t>(totalBytes));
        std::memcpy(result.pixels.data(), image.pixels.data(),
            static_cast<size_t>(result.levels.front().size));

        const auto srgbToLinear = [](float value)
        {
            return value <= 0.04045f ? value / 12.92f :
                std::pow((value + 0.055f) / 1.055f, 2.4f);
        };
        const auto linearToSrgb = [](float value)
        {
            value = std::clamp(value, 0.0f, 1.0f);
            return value <= 0.0031308f ? value * 12.92f :
                1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
        };

        for (uint32_t mip = 1; mip < mipCount; ++mip)
        {
            const ImageMipLevel& srcLevel = result.levels[mip - 1u];
            const ImageMipLevel& dstLevel = result.levels[mip];
            const std::byte* src = result.pixels.data() + srcLevel.offset;
            std::byte* dst = result.pixels.data() + dstLevel.offset;
            for (uint32_t y = 0; y < dstLevel.height; ++y)
            for (uint32_t x = 0; x < dstLevel.width; ++x)
            for (uint32_t channel = 0; channel < image.channels; ++channel)
            {
                float sum = 0.0f;
                for (uint32_t oy = 0; oy < 2u; ++oy)
                for (uint32_t ox = 0; ox < 2u; ++ox)
                {
                    const uint32_t sx = std::min(x * 2u + ox,
                        srcLevel.width - 1u);
                    const uint32_t sy = std::min(y * 2u + oy,
                        srcLevel.height - 1u);
                    const uint64_t index =
                        (static_cast<uint64_t>(sy) * srcLevel.width + sx) *
                            image.channels + channel;
                    float value = 0.0f;
                    if (componentBytes == sizeof(float))
                        std::memcpy(&value, src + index * sizeof(float),
                            sizeof(float));
                    else
                        value = std::to_integer<uint8_t>(src[index]) / 255.0f;
                    if (image.srgb && componentBytes == 1u && channel < 3u)
                        value = srgbToLinear(value);
                    sum += value;
                }
                float value = sum * 0.25f;
                if (image.srgb && componentBytes == 1u && channel < 3u)
                    value = linearToSrgb(value);
                const uint64_t index =
                    (static_cast<uint64_t>(y) * dstLevel.width + x) *
                        image.channels + channel;
                if (componentBytes == sizeof(float))
                    std::memcpy(dst + index * sizeof(float), &value,
                        sizeof(float));
                else
                    dst[index] = static_cast<std::byte>(std::clamp(
                        static_cast<int>(value * 255.0f + 0.5f), 0, 255));
            }
        }
        return result;
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

    inline bool updateGpuOcclusionHistory(
        GpuOcclusionHistoryState& history,
        const SceneRenderView& scene,
        uint32_t width,
        uint32_t height,
        const glm::mat4& currentView,
        const glm::mat4& currentProjection,
        GpuFrameData& frameData)
    {
        const glm::mat4 currentViewProjection =
            currentProjection * currentView;
        const glm::mat4 inverseCurrentView = glm::inverse(currentView);
        const glm::mat4 inversePreviousView = glm::inverse(history.view);
        const glm::vec3 currentForward =
            glm::normalize(-glm::vec3(inverseCurrentView[2]));
        const glm::vec3 previousForward =
            glm::normalize(-glm::vec3(inversePreviousView[2]));

        constexpr float kProjectionEpsilon = 1.0e-5f;
        constexpr float kMaxCameraTranslation = 0.5f;
        constexpr float kMinForwardDot = 0.9902681f; // cos(8 degrees)

        const bool projectionStable =
            std::fabs(history.nearPlane - scene.camera.nearPlane) <=
                kProjectionEpsilon &&
            std::fabs(history.farPlane - scene.camera.farPlane) <=
                kProjectionEpsilon &&
            std::fabs(
                history.verticalFovRadians -
                scene.camera.verticalFovRadians) <= kProjectionEpsilon &&
            std::fabs(history.aspectRatio - scene.camera.aspectRatio) <=
                kProjectionEpsilon;
        const bool cameraContinuous =
            glm::distance(history.cameraPosition, scene.camera.position) <=
                kMaxCameraTranslation &&
            glm::dot(previousForward, currentForward) >= kMinForwardDot;
        const bool reliable =
            history.valid &&
            history.width == width &&
            history.height == height &&
            history.sceneVersion == scene.sceneVersion &&
            projectionStable &&
            cameraContinuous;

        frameData.previousView =
            history.valid ? history.view : currentView;
        frameData.previousViewProjection =
            history.valid ? history.viewProjection : currentViewProjection;
        frameData.occlusionConfig = glm::vec4(
            history.valid ? history.nearPlane : scene.camera.nearPlane,
            1.15f,
            4.0f,
            0.002f);

        history.view = currentView;
        history.viewProjection = currentViewProjection;
        history.cameraPosition = scene.camera.position;
        history.nearPlane = scene.camera.nearPlane;
        history.farPlane = scene.camera.farPlane;
        history.verticalFovRadians = scene.camera.verticalFovRadians;
        history.aspectRatio = scene.camera.aspectRatio;
        history.width = width;
        history.height = height;
        history.sceneVersion = scene.sceneVersion;
        history.valid = true;
        return reliable;
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
