#pragma once

#include <cstdint>
#include <string>

#include <entt/entt.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "ic/core/asset_manager.h"

namespace ic
{
    struct EntityIdComponent
    {
        uint64_t id = 0;
    };

    struct NameComponent
    {
        std::string name;
    };

    struct ActiveComponent
    {
        bool active = true;
    };

    struct TransformComponent
    {
        glm::vec3 translation = glm::vec3(0.0f);
        glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        glm::vec3 scale = glm::vec3(1.0f);

        glm::mat4 local = glm::mat4(1.0f);
        glm::mat4 world = glm::mat4(1.0f);

        uint32_t localVersion = 0;
        uint32_t worldVersion = 0;
        bool dirty = true;
    };

    struct HierarchyComponent
    {
        entt::entity parent = entt::null;
        entt::entity firstChild = entt::null;
        entt::entity nextSibling = entt::null;
        entt::entity previousSibling = entt::null;

        uint16_t depth = 0;
    };

    struct DirtyTransformTag
    {
    };

    struct StaticTag
    {
    };

    struct DynamicTag
    {
    };

    struct ModelComponent
    {
        AssetHandle model = {};
        uint32_t flags = 0;
    };

    struct MaterialOverrideComponent
    {
        AssetHandle material = {};
    };

    struct CameraComponent
    {
        float verticalFovRadians = glm::radians(60.0f);
        float nearPlane = 0.1f;
        float farPlane = 1000.0f;

        bool primary = false;
    };

    struct FlyCameraControllerComponent
    {
        float moveSpeed = 5.0f;
        float fastMoveMultiplier = 4.0f;
        float mouseSensitivity = 0.0025f;

        float yaw = 0.0f;
        float pitch = 0.0f;

        bool captureMouse = true;
    };

    enum class LightType : uint8_t
    {
        Directional = 0,
        Point,
        Spot
    };

    struct LightComponent
    {
        LightType type = LightType::Point;

        glm::vec3 color = glm::vec3(1.0f);
        float intensity = 1.0f;

        float range = 10.0f;
        float innerConeRadians = glm::radians(15.0f);
        float outerConeRadians = glm::radians(30.0f);
        float padding = 0.0f;
    };

    struct VelocityComponent
    {
        glm::vec3 linear = glm::vec3(0.0f);
        glm::vec3 angular = glm::vec3(0.0f);
    };
}
