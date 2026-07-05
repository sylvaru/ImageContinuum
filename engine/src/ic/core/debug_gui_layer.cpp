// ic/core/debug_gui_layer.cpp
#include "ic/common/ic_pch.h"
#include "ic/core/debug_gui_layer.h"
#include "ic/core/app_base.h"
#include "ic/core/frame_context.h"
#include "ic/renderer/renderer.h"
#include "ic/scene/scene.h"
#include "ic/scene/scene_manager.h"

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

        m_renderer = ctx.services ? ctx.services->renderer : nullptr;
        m_sceneManager = ctx.services ? ctx.services->sceneManager : nullptr;
    }

    void DebugGuiLayer::onRender([[maybe_unused]] float alpha)
    {
        constexpr ImGuiWindowFlags flags =
            ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoCollapse;

        ImGui::SetNextWindowPos(ImVec2(12.0f, 12.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(220.0f, 100.0f), ImGuiCond_FirstUseEver);

        if (ImGui::Begin("Debug", nullptr, flags))
        {
            ImGui::Text("FPS: %.1f", m_displayFps);
            ImGui::Text("Frame: %.3f ms", m_displayFrameTimeMs);

            if (m_renderer)
            {
                bool vsync = m_renderer->vsyncEnabled();
                if (ImGui::Checkbox("VSync", &vsync))
                {
                    m_renderer->setVsyncEnabled(vsync);
                }
            }

            if (m_sceneManager)
            {
                if (Scene* scene = m_sceneManager->activeScene())
                {
                    EnvironmentSettings settings =
                        scene->environmentSettings();
                    bool changed = false;

                    ImGui::Separator();
                    changed |=
                        ImGui::Checkbox(
                            "Environment",
                            &settings.enabled);
                    changed |=
                        ImGui::SliderFloat(
                            "Env Intensity",
                            &settings.intensity,
                            0.0f,
                            8.0f);
                    changed |=
                        ImGui::SliderFloat(
                            "Skybox Exposure",
                            &settings.skyboxExposure,
                            0.0f,
                            4.0f);
                    changed |=
                        ImGui::SliderFloat(
                            "Path Env Exposure",
                            &settings.pathTraceExposure,
                            0.0f,
                            4.0f);
                    changed |=
                        ImGui::SliderFloat(
                            "Tonemap Exposure",
                            &settings.tonemapExposure,
                            0.0f,
                            4.0f);

                    if (changed)
                    {
                        scene->setEnvironmentSettings(settings);
                    }
                }
            }
        }

        ImGui::End();
    }
}
