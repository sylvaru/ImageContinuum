// image_continuum/core/event_frame_buffer.cpp
#include "image_continuum/core/event_frame_buffer.h"
#include "image_continuum/core/frame.h"

namespace ic
{
    // Called from producer threads
    void EventFrameBuffer::push(EventChannel ch, const Event& e)
    {
        m_frames[m_write].get(ch).push(e);
    }

    // Called at frame start (game thread)
    void EventFrameBuffer::beginFrame(FrameContext& ctx)
    {
        ctx.frameIndex++;

        const int read = m_write;
        m_write ^= 1;

        ctx.eventFrame = &m_frames[read];
    }

}