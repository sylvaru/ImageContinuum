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

        FrameArena* arena = nullptr;
        EventFrame* eventFrame = nullptr;
        AppServices* services = nullptr;


        uint32_t gpuFrameIndex(uint32_t framesInFlight) const
        {
            return static_cast<uint32_t>(frameIndex % framesInFlight);
        }
    };


}