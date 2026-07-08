// ic/core/frame_pipeline.cpp
#include "ic/common/ic_pch.h"
#include "ic/core/frame_pipeline.h"
#include "ic/core/app_base.h"
#include "ic/util/profiler.h"
#include "ic/core/event_frame_buffer.h"
#include "ic/core/memory/frame_arena.h"
#include "ic/core/frame_context.h"


#include <spdlog/spdlog.h>

namespace ic
{
    FramePipeline::FramePipeline(AppBase& app)
        : m_app(app)
    {
    }

    void FramePipeline::execute(FrameContext& frame)
    {
        runInput(frame);
        runSimulation(frame);
    }

    void FramePipeline::runInput(FrameContext& frame)
    {
        ZoneScopedN("runInput");

        auto& f = *frame.eventFrame;

        for (size_t i = 0; i < kEventChannelCount; ++i)
        {
            EventChannel ch = static_cast<EventChannel>(i);

            f.channels[i].drain(
                [this, ch](Event& e)
                {
                    m_app.dispatchEvent(ch, e);
                });
        }
    }

    void FramePipeline::runSimulation(FrameContext& frame)
    {
        ZoneScopedN("runSimulation");
        m_app.onUpdate(frame.deltaTime);
        m_app.layerStack().updateAll(frame);
    }
}