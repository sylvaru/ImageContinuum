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
        m_sceneManager = ctx.services ? ctx.services->sceneManager : nullptr;
    }

    // Scene authoring controls only.
    //
    // Everything that describes the renderer -- frame/GPU timing, async
    // compute, queues, the frame graph, pass timings, resources, culling and
    // the Hi-Z pyramid -- now lives in the single "Renderer Diagnostics" window
    // owned by the Renderer, which is where the data already is. What is left
    // here changes the scene rather than reporting on it, so it stays separate.
    void DebugGuiLayer::onRender([[maybe_unused]] float alpha)
    {
        if (!m_sceneManager)
        {
            return;
        }

        Scene* scene = m_sceneManager->activeScene();
        if (!scene)
        {
            return;
        }

        constexpr ImGuiWindowFlags flags = ImGuiWindowFlags_NoSavedSettings;

        ImGui::SetNextWindowPos(ImVec2(12.0f, 644.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(300.0f, 180.0f), ImGuiCond_FirstUseEver);

        if (ImGui::Begin("Scene", nullptr, flags))
        {
            EnvironmentSettings settings = scene->environmentSettings();
            bool changed = false;

            changed |= ImGui::Checkbox("Environment", &settings.enabled);
            changed |= ImGui::SliderFloat(
                "Env Intensity", &settings.intensity, 0.0f, 8.0f);
            changed |= ImGui::SliderFloat(
                "Skybox Exposure", &settings.skyboxExposure, 0.0f, 4.0f);
            changed |= ImGui::SliderFloat(
                "Path Env Exposure", &settings.pathTraceExposure, 0.0f, 4.0f);
            changed |= ImGui::SliderFloat(
                "Tonemap Exposure", &settings.tonemapExposure, 0.0f, 4.0f);

            if (changed)
            {
                scene->setEnvironmentSettings(settings);
            }
        }

        ImGui::End();
    }
}
