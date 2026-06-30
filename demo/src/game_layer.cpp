// demo/src/game_layer.cpp
#include "game_layer.h"

#include "ic/core/app_base.h"
#include "ic/core/frame_context.h"
#include "ic/scene/scene_manager.h"

#include <spdlog/spdlog.h>

void GameLayer::onAttach(ic::AppServices& services)
{
    if (!services.sceneManager)
    {
        return;
    }

    ic::SceneLoadOptions options{};
    options.mode = ic::SceneLoadMode::ReplaceActive;
    options.loadReferencedAssets = true;
    options.makeActive = true;

    m_scene = services.sceneManager->loadSceneAsync(
        "demo/res/cornell_scene.toml",
        options);

    spdlog::info("[Demo] Queued scene load");
}

void GameLayer::onUpdate(
	[[maybe_unused]] ic::FrameContext& ctx) {
    // Gameplay systems will move here as the demo scene grows.
}

void GameLayer::onRender([[maybe_unused]] float alpha) {
	/* Submit flat data buffers */ 
}

