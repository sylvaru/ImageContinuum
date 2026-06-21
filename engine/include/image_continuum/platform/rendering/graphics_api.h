// graphics_api.h
#pragma once
#include "image_continuum/interface/window.h"
#include "image_continuum/interface/renderer_backend.h"

namespace ic
{

	std::unique_ptr<RendererBackend>
		createRenderer(Window& window, const RendererSpecification& spec);
}