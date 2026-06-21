// image_continuum/core/event_frame_buffer.h
#pragma once
#include "event_queue.h"
#include <array>
#include <atomic>
#include "image_continuum/interface/events.h"

namespace ic
{
    struct FrameContext;

    struct EventFrame
    {
        std::array<EventQueue, kEventChannelCount> channels;

        EventQueue& get(EventChannel c)
        {
            return channels[static_cast<size_t>(c)];
        }
    };



    // Frame local event storage
    class EventFrameBuffer
    {
    public:
        static constexpr size_t kFrameCount = 2;

        EventFrameBuffer() = default;

        // Called from producer threads (GLFW/input/etc)
        void push(EventChannel ch, const Event& e);

        // Called at frame start (game thread)
        void beginFrame(FrameContext& ctx);

    private:

        std::array<EventFrame, kFrameCount> m_frames;

        int m_write = 0;
    };
}