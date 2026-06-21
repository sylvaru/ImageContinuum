#pragma once
#include "image_continuum/common/ic_pch.h"
#include "image_continuum/platform/rendering/graphics_api.h"


namespace ic
{


    std::unique_ptr<RendererBackend>
        createRenderer(
            Window& window,
            const RendererSpecification& spec)
    {
        (void)window, spec;

        switch (spec.graphicsAPI)
        {
        case GraphicsAPI::Vulkan:
            //return std::make_unique<VulkanBackend>(window, spec);
            return nullptr;
        case GraphicsAPI::DX12:
            //return std::make_unique<DX12Backend>(window, spec);
            return nullptr;

        default:
            return nullptr;
        }
    }
}