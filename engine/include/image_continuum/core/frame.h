// image_continuum/core/frame.h
#pragma once
#include <cstdint>

namespace ic
{
    struct EventFrame;

    struct FrameContext
    {
        // frame identity
        uint64_t frameIndex = 0;

        // timing
        float deltaTime = 0.0f;
        float timeSinceStart = 0.0f;

        float interpolationAlpha = 1.0f;

        // events (snapshot for this frame)
        EventFrame* eventFrame;

        // job system hook (future)
        //JobSystem* jobs = nullptr;

        // renderer backend hook (DX12/Vulkan abstraction)
        
        //Renderer* renderer = nullptr;

        // GPU frame index (swapchain / frame-in-flight)
        uint32_t gpuFrameIndex = 0;
    };


}