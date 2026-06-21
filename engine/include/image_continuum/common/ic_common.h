// image_continuum/common/ic_common.h
#pragma once


namespace ic
{
    struct FrameData {
        //alignas(256) Matrix4 model;
        //alignas(256) Matrix4 viewProj;
        //alignas(16)  Vector4 cameraPos;
        alignas(4)   uint32_t materialIndex;
    };
    

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
} 