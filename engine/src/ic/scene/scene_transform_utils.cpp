#include "ic/common/ic_pch.h"
#include "ic/scene/scene_transform_utils.h"

namespace ic
{
    void detachFromParent(entt::registry& registry, entt::entity child)
    {
        auto* childHierarchy = registry.try_get<HierarchyComponent>(child);
        if (!childHierarchy)
        {
            return;
        }

        const entt::entity parent = childHierarchy->parent;
        if (parent != entt::null)
        {
            auto* parentHierarchy = registry.try_get<HierarchyComponent>(parent);
            if (parentHierarchy && parentHierarchy->firstChild == child)
            {
                parentHierarchy->firstChild = childHierarchy->nextSibling;
            }
        }

        if (childHierarchy->previousSibling != entt::null)
        {
            auto& previous =
                registry.get<HierarchyComponent>(childHierarchy->previousSibling);
            previous.nextSibling = childHierarchy->nextSibling;
        }

        if (childHierarchy->nextSibling != entt::null)
        {
            auto& next =
                registry.get<HierarchyComponent>(childHierarchy->nextSibling);
            next.previousSibling = childHierarchy->previousSibling;
        }

        childHierarchy->parent = entt::null;
        childHierarchy->nextSibling = entt::null;
        childHierarchy->previousSibling = entt::null;
        childHierarchy->depth = 0;

        markTransformDirty(registry, child);
    }

    void setParent(
        entt::registry& registry,
        entt::entity child,
        entt::entity newParent)
    {
        if (child == entt::null || child == newParent || !registry.valid(child))
        {
            return;
        }

        if (newParent != entt::null && !registry.valid(newParent))
        {
            return;
        }

        if (!registry.all_of<HierarchyComponent>(child))
        {
            registry.emplace<HierarchyComponent>(child);
        }

        if (newParent != entt::null &&
            !registry.all_of<HierarchyComponent>(newParent))
        {
            registry.emplace<HierarchyComponent>(newParent);
        }

        detachFromParent(registry, child);

        auto& childHierarchy = registry.get<HierarchyComponent>(child);
        childHierarchy.parent = newParent;

        if (newParent != entt::null)
        {
            auto& parentHierarchy = registry.get<HierarchyComponent>(newParent);

            childHierarchy.nextSibling = parentHierarchy.firstChild;
            if (parentHierarchy.firstChild != entt::null)
            {
                auto& oldFirst =
                    registry.get<HierarchyComponent>(parentHierarchy.firstChild);
                oldFirst.previousSibling = child;
            }

            parentHierarchy.firstChild = child;
            childHierarchy.depth =
                static_cast<uint16_t>(parentHierarchy.depth + 1);
        }

        updateHierarchyDepthRecursive(registry, child, childHierarchy.depth);
        markTransformDirty(registry, child);
    }

    void updateHierarchyDepthRecursive(
        entt::registry& registry,
        entt::entity entity,
        uint16_t depth)
    {
        auto* hierarchy = registry.try_get<HierarchyComponent>(entity);
        if (!hierarchy)
        {
            return;
        }

        hierarchy->depth = depth;

        entt::entity child = hierarchy->firstChild;
        while (child != entt::null)
        {
            auto* childHierarchy = registry.try_get<HierarchyComponent>(child);
            const entt::entity next =
                childHierarchy ? childHierarchy->nextSibling : entt::null;

            updateHierarchyDepthRecursive(
                registry,
                child,
                static_cast<uint16_t>(depth + 1));

            child = next;
        }
    }
}
