#pragma once

#include "ic/scene/scene.h"
#include "ic/scene/scene_entity.h"

namespace ic
{
    template <typename Component, typename... Args>
    Component& SceneEntity::addComponent(Args&&... args)
    {
        return m_scene->registry().emplace<Component>(
            m_entity,
            std::forward<Args>(args)...);
    }

    template <typename Component, typename... Args>
    Component& SceneEntity::addOrReplaceComponent(Args&&... args)
    {
        return m_scene->registry().emplace_or_replace<Component>(
            m_entity,
            std::forward<Args>(args)...);
    }

    template <typename Component>
    void SceneEntity::removeComponent()
    {
        m_scene->registry().remove<Component>(m_entity);
    }

    template <typename Component>
    bool SceneEntity::hasComponent() const
    {
        return m_scene->registry().all_of<Component>(m_entity);
    }

    template <typename Component>
    Component& SceneEntity::getComponent()
    {
        return m_scene->registry().get<Component>(m_entity);
    }

    template <typename Component>
    const Component& SceneEntity::getComponent() const
    {
        return m_scene->registry().get<Component>(m_entity);
    }
}
