#include "image_continuum/core/frame_executor.h"
#include "image_continuum/core/app_base.h"

namespace ic
{
    FrameExecutor::FrameExecutor(AppBase& app)
        : m_app(app)
    {
    }

    void FrameExecutor::execute(FrameContext& frame)
    {
        runInput(frame);
        runSimulation(frame);
        runRenderPrep(frame);
        runRenderSubmit(frame);
    }

    void FrameExecutor::runInput(FrameContext& frame)
    {
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

    void FrameExecutor::runSimulation(FrameContext& frame)
    {
        m_app.onUpdate(frame.deltaTime);
        m_app.layerStack().updateAll(frame.deltaTime);
    }

    void FrameExecutor::runRenderPrep([[maybe_unused]] FrameContext& frame)
    {
        // build render graph, cull, prepare GPU data
    }

    void FrameExecutor::runRenderSubmit(FrameContext& frame)
    {
        m_app.layerStack().renderAll(frame.interpolationAlpha);
    }
}