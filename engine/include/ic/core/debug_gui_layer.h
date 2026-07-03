// ic/core/debug_gui_layer.h
#pragma once

#include "ic/interface/layer.h"

namespace ic
{
    struct FrameContext;
    class Renderer;

    class DebugGuiLayer final : public Layer
    {
    public:
        DebugGuiLayer();

        void onUpdate(FrameContext& ctx) override;
        void onRender(float alpha) override;

    private:
        float m_displayFrameTimeMs = 0.0f;
        float m_displayFps = 0.0f;
        float m_sampleElapsed = 0.0f;
        uint32_t m_sampleFrames = 0;
        Renderer* m_renderer = nullptr;
    };
}
