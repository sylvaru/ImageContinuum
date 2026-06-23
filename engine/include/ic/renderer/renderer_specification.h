// ic/rendering/renderer_specification.h
#pragma once

namespace ic
{

    enum class GraphicsAPI : uint8_t
    {
        Vulkan,
        DX12
    };


    struct RendererSpecification
    {
        GraphicsAPI graphicsAPI = GraphicsAPI::Vulkan;
        bool        enableValidation = true;
        bool        enableImGui = true;
        uint32_t    framesInFlight = 2;
    };

}