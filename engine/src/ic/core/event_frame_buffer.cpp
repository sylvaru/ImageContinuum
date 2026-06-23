// ic/core/event_frame_buffer.cpp
#include "ic/common/ic_pch.h"
#include "ic/core/event_frame_buffer.h"
#include "ic/core/frame.h"

namespace ic
{
    // Called from producer threads
    void EventFrameBuffer::ingest(EventQueue& globalQueue)
    {
        Event e;

        while (globalQueue.tryPop(e))
        {
            auto ch = channelForEvent(e.type);
            m_frames[m_write].get(ch).push(e);
        }
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