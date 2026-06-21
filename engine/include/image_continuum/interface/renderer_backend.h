// image_continuum/interface/render_backend.h
#pragma once

namespace ic
{
    enum class GraphicsAPI
    {
        Vulkan,
        DX12
    };

    struct RendererSpecification
    {
        GraphicsAPI graphicsAPI = GraphicsAPI::Vulkan;

        bool enableValidation = true;

        bool enableImGui = true;

        uint32_t framesInFlight = 2;

    };

    struct RendererBackend
    {
        virtual ~RendererBackend() = default;

        virtual void beginFrame() = 0;
        virtual void endFrame() = 0;
    };

}