// ic/core/memory/frame_memory_types.h
#pragma once


namespace ic
{
    enum class FrameRegion : uint8_t
    {
        Simulation,
        RenderPrep,
        GPUUpload,
        Temp,
        Count
    };

    enum class FrameTag : uint16_t
    {
        Unknown,
        ECS,
        Render,
        GPU,
        Temp,
    };
}