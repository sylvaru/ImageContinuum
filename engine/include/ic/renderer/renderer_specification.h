// ic/rendering/renderer_specification.h
#pragma once

namespace ic
{

    enum class RendererBackendType : uint8_t
    {
        Vulkan,
        DX12
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
        bool                enableImGui = true;
        uint32_t            framesInFlight = 2;
    };

}