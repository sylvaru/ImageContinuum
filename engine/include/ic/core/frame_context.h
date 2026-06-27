// ic/core/frame.h
#pragma once
#include <cstdint>

namespace ic
{
    struct AppServices;
    struct EventFrame;

    class FrameArena;    
    

    struct FrameContext
    {
        uint64_t frameIndex;
        float deltaTime;
        float timeSinceStart;
        float interpolationAlpha;
        uint32_t gpuFrameIndex;

        FrameArena* arena = nullptr;
        EventFrame* eventFrame = nullptr;
        AppServices* services = nullptr;
    };


}