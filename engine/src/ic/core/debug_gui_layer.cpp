// ic/core/debug_gui_layer.cpp
#include "ic/common/ic_pch.h"
#include "ic/core/debug_gui_layer.h"
#include "ic/core/frame_context.h"

#include <imgui.h>

namespace ic
{
    DebugGuiLayer::DebugGuiLayer()
        : Layer({ .debugName = "DebugGuiLayer" })
    {
    }

    void DebugGuiLayer::onUpdate(FrameContext& ctx)
    {
        m_sampleElapsed += ctx.deltaTime;
        ++m_sampleFrames;

        constexpr float kSampleInterval = 0.25f;
        if (m_sampleElapsed >= kSampleInterval)
        {
            m_displayFps =
                static_cast<float>(m_sampleFrames) / m_sampleElapsed;
            m_displayFrameTimeMs =
                (m_sampleElapsed / static_cast<float>(m_sampleFrames)) *
                1000.0f;

            m_sampleElapsed = 0.0f;
            m_sampleFrames = 0;
        }
    }

    void DebugGuiLayer::onRender([[maybe_unused]] float alpha)
    {
        constexpr ImGuiWindowFlags flags =
            ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoCollapse;

        ImGui::SetNextWindowPos(ImVec2(12.0f, 12.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(180.0f, 76.0f), ImGuiCond_FirstUseEver);

        if (ImGui::Begin("Debug", nullptr, flags))
        {
            ImGui::Text("FPS: %.1f", m_displayFps);
            ImGui::Text("Frame: %.3f ms", m_displayFrameTimeMs);
        }

        ImGui::End();
    }
}
