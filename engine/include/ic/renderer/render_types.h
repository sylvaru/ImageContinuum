// ic/rendering/render_types.h
#pragma once


namespace ic 
{
    enum class QueueType : uint8_t
    {
        Graphics,
        Compute,
        Transfer
    };

    // A single, tightly packed indirect draw command (12 bytes)
    // Fed to GPU via DrawIndexedIndirect or ExecuteIndirect
    struct DrawCommand 
    {
        uint32_t pipelineIndex;
        uint32_t materialIndex; // Index into a global SSBO (Bindless)
        uint32_t meshIndex;     // Index into global vertex/index buffer (BDA)
    };

    // GPU Ring buffer
}