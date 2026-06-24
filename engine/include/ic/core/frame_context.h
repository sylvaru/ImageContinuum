// ic/core/frame.h
#pragma once
#include <cstdint>

namespace ic
{
    struct EventFrame;
    struct Input;
    class FrameArena;
    class JobSystem;
    

    struct FrameContext
    {
        uint64_t frameIndex;

        float deltaTime;
        float timeSinceStart;
        float interpolationAlpha;

        uint32_t gpuFrameIndex;
        
        EventFrame* eventFrame;
        Input* input;
        JobSystem* jobs;
        //RendererBackend* renderer;

        FrameArena* arena;
    };


}