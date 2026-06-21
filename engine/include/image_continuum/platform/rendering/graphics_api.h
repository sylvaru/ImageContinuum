// graphics_api.h
#pragma once
#include "image_continuum/interface/window.h"
#include "image_continuum/interface/renderer_backend.h"

namespace ic
{

    struct FrameData {
        //alignas(256) Matrix4 model;
        //alignas(256) Matrix4 viewProj;
        //alignas(16)  Vector4 cameraPos;
        alignas(4)   uint32_t materialIndex;
    };


	std::unique_ptr<RendererBackend>
		createRenderer(Window& window, const RendererSpecification& spec);
}


/*
Bad:

VkBuffer
VkImage
VkDescriptorSet
VkCommandBuffer

escaping into engine code.

Good:

BufferHandle
TextureHandle
PipelineHandle
CommandList

The render graph, ECS, scene systems, culling, streaming, etc should not know Vulkan exists.

The renderer backend owns Vulkan.


*/