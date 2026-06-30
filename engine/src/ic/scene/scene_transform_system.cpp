#include "ic/common/ic_pch.h"
#include "ic/scene/scene_transform_system.h"

#include "ic/scene/scene_components.h"
#include "ic/scene/scene_transform_utils.h"

namespace ic
{
    void SceneTransformSystem::update(entt::registry& registry)
    {
        auto roots = registry.view<TransformComponent, HierarchyComponent>();

        for (entt::entity entity : roots)
        {
            const auto& hierarchy = roots.get<HierarchyComponent>(entity);
            if (hierarchy.parent != entt::null)
            {
                continue;
            }

            updateEntityRecursive(registry, entity, glm::mat4(1.0f), false);
        }

        registry.clear<DirtyTransformTag>();
    }

    void SceneTransformSystem::updateEntityRecursive(
        entt::registry& registry,
        entt::entity entity,
        const glm::mat4& parentWorld,
        bool parentDirty)
    {
        auto* transform = registry.try_get<TransformComponent>(entity);
        auto* hierarchy = registry.try_get<HierarchyComponent>(entity);
        if (!transform || !hierarchy)
        {
            return;
        }

        const bool dirty =
            parentDirty ||
            transform->dirty ||
            registry.all_of<DirtyTransformTag>(entity);

        if (dirty)
        {
            transform->local = composeTransform(
                transform->translation,
                transform->rotation,
                transform->scale);
            transform->world = parentWorld * transform->local;
            transform->dirty = false;
            ++transform->worldVersion;
        }

        entt::entity child = hierarchy->firstChild;
        while (child != entt::null)
        {
            auto* childHierarchy = registry.try_get<HierarchyComponent>(child);
            const entt::entity next =
                childHierarchy ? childHierarchy->nextSibling : entt::null;

            updateEntityRecursive(registry, child, transform->world, dirty);
            child = next;
        }
    }
}
