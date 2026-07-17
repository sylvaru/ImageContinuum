// ic/core/debug_gui_layer.h
#pragma once

#include "ic/interface/layer.h"

namespace ic
{
    struct FrameContext;
    class SceneManager;

    // Scene authoring controls. Renderer telemetry and controls live in the
    // Renderer's own consolidated "Renderer Diagnostics" window.
    class DebugGuiLayer final : public Layer
    {
    public:
        DebugGuiLayer();

        void onUpdate(FrameContext& ctx) override;
        void onRender(float alpha) override;

    private:
        SceneManager* m_sceneManager = nullptr;
    };
}
