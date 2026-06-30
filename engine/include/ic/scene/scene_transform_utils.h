#pragma once

#include <cstdint>

#include <entt/entt.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include "ic/scene/scene_components.h"

namespace ic
{
    inline glm::mat4 composeTransform(
        const glm::vec3& translation,
        const glm::quat& rotation,
        const glm::vec3& scale)
    {
        return glm::translate(glm::mat4(1.0f), translation) *
               glm::mat4_cast(rotation) *
               glm::scale(glm::mat4(1.0f), scale);
    }

    inline void markTransformDirty(entt::registry& registry, entt::entity entity)
    {
        if (!registry.valid(entity))
        {
            return;
        }

        auto* transform = registry.try_get<TransformComponent>(entity);
        if (!transform)
        {
            return;
        }

        transform->dirty = true;
        ++transform->localVersion;

        if (!registry.all_of<DirtyTransformTag>(entity))
        {
            registry.emplace<DirtyTransformTag>(entity);
        }
    }

    void setParent(
        entt::registry& registry,
        entt::entity child,
        entt::entity newParent);

    void detachFromParent(
        entt::registry& registry,
        entt::entity child);

    void updateHierarchyDepthRecursive(
        entt::registry& registry,
        entt::entity entity,
        uint16_t depth);
}
