// ic/rendering/renderer_specification.h
#pragma once
#include <cstdint>
#include <filesystem>
#include "render_types.h"

namespace ic
{
    struct SwapchainInfo
    {
        uint32_t width = 0;
        uint32_t height = 0;
        TextureFormat format = TextureFormat::Unknown;
        uint32_t imageCount = 0;
        bool valid = false;
    };

    struct EnvironmentSettings
    {
        bool enabled = true;
        float intensity = 1.0f;
        float skyboxExposure = 0.9f;
        float pathTraceExposure = 0.25f;
        float tonemapExposure = 1.0f;
        uint32_t cubemapSize = 512;
    };

    struct RendererSettings
    {
        bool vsync = true;
        // Uncapped by default. Vsync remains a separate presentation control.
        float targetFps = 0.0f;
        bool gpuOcclusion = true;
        GpuCullDebugMode gpuCullDebugMode = GpuCullDebugMode::Off;
        EnvironmentSettings environment;
    };

    enum class RendererBackendType : uint8_t
    {
        Vulkan,
#ifdef _WIN32
        DX12
#endif
    };

    enum class RenderPathType : uint8_t
    {
        Forward,
        ClusteredForward, // Preferred
        Deferred,
        PathTraced
    };


    struct RendererSpecification
    {
        RendererBackendType backendType;
        RenderPathType      pathType;
        bool                enableValidation = true;
        bool                useDebugGui = true;
        RendererSettings    settings;
        uint32_t            framesInFlight = 2;
        std::filesystem::path pipelineLibraryPath;
    };

}
