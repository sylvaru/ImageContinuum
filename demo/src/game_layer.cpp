// demo/src/game_layer.cpp
#include "game_layer.h"

#include "ic/core/app_base.h"
#include "ic/core/frame_context.h"
#include "ic/scene/scene_manager.h"
#include "ic/scene/scene_components.h"
#include "ic/scene/scene_transform_utils.h"

#include <cstdlib>
#include <cmath>

namespace
{
    bool environmentPresent(const char* name)
    {
#ifdef _WIN32
        char* value = nullptr;
        size_t length = 0;
        const bool present = _dupenv_s(&value, &length, name) == 0 &&
            value != nullptr;
        std::free(value);
        return present;
#else
        return std::getenv(name) != nullptr;
#endif
    }
}
#include <spdlog/spdlog.h>

GameLayer::GameLayer(std::filesystem::path scenePath)
    : m_scenePath(std::move(scenePath))
{
}

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
        m_scenePath,
        options);
    m_cameraSweepEnabled = environmentPresent("IC_GI_CAMERA_SWEEP");
    m_dynamicSweepEnabled = environmentPresent("IC_GI_DYNAMIC_INSTANCE_SWEEP");

    spdlog::info(
        "[Demo] Queued scene load: {}",
        m_scenePath.string());
}

void GameLayer::onUpdate(ic::FrameContext& ctx)
{
    if ((!m_cameraSweepEnabled && !m_dynamicSweepEnabled) || !ctx.services ||
        !ctx.services->sceneManager)
        return;
    entt::registry* registry = ctx.services->sceneManager->activeRegistry();
    if (!registry) return;
    if (m_cameraSweepEnabled)
    {
        auto cameras = registry->view<ic::CameraComponent,
            ic::TransformComponent>();
        for (const entt::entity entity : cameras)
        {
            const auto& camera = cameras.get<ic::CameraComponent>(entity);
            if (!camera.primary) continue;
            auto& transform = cameras.get<ic::TransformComponent>(entity);
            if (!m_cameraSweepInitialized)
            {
                m_cameraSweepOrigin = transform.translation;
                m_cameraSweepInitialized = true;
            }
            const float phase = ctx.timeSinceStart * 0.65f;
            transform.translation = m_cameraSweepOrigin + glm::vec3(
                std::sin(phase) * 0.8f, std::sin(phase * 0.5f) * 0.12f,
                std::cos(phase) * 0.25f);
            transform.dirty = true;
            ic::markTransformDirty(*registry, entity);
            break;
        }
    }
    if (m_dynamicSweepEnabled)
    {
        auto models = registry->view<ic::NameComponent, ic::ModelComponent,
            ic::TransformComponent>();
        for (const entt::entity entity : models)
        {
            if (models.get<ic::NameComponent>(entity).name.find("Helmet") ==
                std::string::npos)
                continue;
            auto& transform = models.get<ic::TransformComponent>(entity);
            if (!m_dynamicSweepInitialized)
            {
                m_dynamicSweepOrigin = transform.translation;
                m_dynamicSweepInitialized = true;
            }
            transform.translation = m_dynamicSweepOrigin + glm::vec3(
                std::sin(ctx.timeSinceStart * 1.1f) * 0.6f, 0.0f, 0.0f);
            transform.dirty = true;
            ic::markTransformDirty(*registry, entity);
            break;
        }
    }
}

void GameLayer::onRender([[maybe_unused]] float alpha)
{

}

