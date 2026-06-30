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
        uint64_t frameIndex = 0;
        float deltaTime = 0.0f;
        float timeSinceStart = 0.0f;
        float interpolationAlpha = 0.0f;

        uint32_t windowWidth = 0;
        uint32_t windowHeight = 0;

        FrameArena* arena = nullptr;
        EventFrame* eventFrame = nullptr;
        AppServices* services = nullptr;


        uint32_t gpuFrameIndex(uint32_t framesInFlight) const
        {
            return static_cast<uint32_t>(frameIndex % framesInFlight);
        }
    };


}
