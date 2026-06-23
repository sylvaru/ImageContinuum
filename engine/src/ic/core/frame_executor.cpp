// ic/core/frame_executor.cpp
#include "ic/common/ic_pch.h"
#include "ic/core/frame_executor.h"
#include "ic/core/app_base.h"
#include "ic/util/profiler.h"
#include "ic/core/event_frame_buffer.h"

namespace ic
{
    FrameExecutor::FrameExecutor(AppBase& app)
        : m_app(app)
    {
    }

    void FrameExecutor::execute([[maybe_unused]] FrameContext& frame)
    {
        runInput(frame);
        runSimulation(frame);
        runRenderPrep(frame);
        runRenderSubmit(frame);
    }

    void FrameExecutor::runInput(FrameContext& frame)
    {
        //ZoneScopedN("runInput");

        auto& f = *frame.eventFrame;

        for (size_t i = 0; i < kEventChannelCount; ++i)
        {
            EventChannel ch = static_cast<EventChannel>(i);

            f.channels[i].drain(
                [this, ch](Event& e)
                {
                    (void)e;
                    m_app.dispatchEvent(ch, e);
                });
        }
    }

    void FrameExecutor::runSimulation(FrameContext& frame)
    {
        //ZoneScopedN("runSimulation");
        m_app.onUpdate(frame.deltaTime);
        m_app.layerStack().updateAll(frame);
    }

    void FrameExecutor::runRenderPrep([[maybe_unused]] FrameContext& frame)
    {
        //ZoneScopedN("runRenderPrep");
        // build render graph, cull, prepare GPU data
    }

    void FrameExecutor::runRenderSubmit([[maybe_unused]] FrameContext& frame)
    {
        //ZoneScopedN("runRenderSubmit");
        m_app.layerStack().renderAll(frame.interpolationAlpha);
    }
}