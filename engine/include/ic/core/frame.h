// ic/core/frame.h
#pragma once
#include <cstdint>

namespace ic
{
    struct EventFrame;
    class JobSystem;
    struct Input;

    struct FrameContext
    {
        uint64_t frameIndex;

        float deltaTime;
        float timeSinceStart;
        float interpolationAlpha;

        uint32_t gpuFrameIndex;
        
        EventFrame* eventFrame;
        Input* input;
        //FrameArena* arena;
        JobSystem* jobs;
        //RendererBackend* renderer;
    };


}