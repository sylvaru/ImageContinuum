#pragma once

#include <cstdint>
#include <span>

#include <entt/entt.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "ic/core/asset_manager.h"
#include "ic/scene/scene_components.h"

namespace ic
{
    struct SceneCameraView
    {
        entt::entity entity = entt::null;

        glm::mat4 view = glm::mat4(1.0f);
        glm::mat4 projection = glm::mat4(1.0f);
        glm::mat4 viewProjection = glm::mat4(1.0f);

        glm::vec3 position = glm::vec3(0.0f);
        float nearPlane = 0.1f;

        float farPlane = 1000.0f;
        float verticalFovRadians = glm::radians(60.0f);
        float aspectRatio = 1.0f;
        uint32_t valid = 0;
    };

    struct SceneModelRenderItem
    {
        entt::entity entity = entt::null;

        AssetHandle model = {};
        AssetHandle materialOverride = {};

        glm::mat4 world = glm::mat4(1.0f);

        uint32_t flags = 0;
    };

    struct SceneLightRenderItem
    {
        entt::entity entity = entt::null;

        LightType type = LightType::Point;

        glm::vec3 position = glm::vec3(0.0f);
        float range = 0.0f;

        glm::vec3 direction = glm::vec3(0.0f, -1.0f, 0.0f);
        float intensity = 1.0f;

        glm::vec3 color = glm::vec3(1.0f);
        float outerConeRadians = 0.0f;

        float innerConeRadians = 0.0f;
        float padding0 = 0.0f;
        float padding1 = 0.0f;
        float padding2 = 0.0f;
    };

    struct SceneRenderView
    {
        SceneCameraView camera = {};

        std::span<const SceneModelRenderItem> models;
        std::span<const SceneLightRenderItem> lights;

        uint64_t sceneVersion = 0;
        uint64_t frameIndex = 0;
    };
}
