// ic/rendering/renderer_specification.h
#pragma once
#include <cstdint>
#include <filesystem>

namespace ic
{
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
        float targetFps = 500.0f;
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
