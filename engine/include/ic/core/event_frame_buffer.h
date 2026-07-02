// ic/core/event_frame_buffer.h
#pragma once
#include "event_queue.h"
#include <array>
#include "ic/core/events.h"

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

    template<size_t FrameCount>
    class EventFrameBuffer
    {
    public:

        EventFrameBuffer() = default;

        void ingest(EventQueue& globalQueue)
        {
            Event e;
            while (globalQueue.tryPop(e))
            {
                auto ch = channelForEvent(e.type);
                m_frames[m_write].get(ch).push(e);
            }
        }

        // Called at frame start (game thread)
        void beginFrame(FrameContext& ctx)
        {
            // Ring buffer indexing
            const size_t read = m_write;
            m_write = (ctx.frameIndex + 1) % FrameCount;
            ctx.eventFrame = &m_frames[read];
        }

    private:

        std::array<EventFrame, FrameCount> m_frames;

        int m_write = 0;
    };
}