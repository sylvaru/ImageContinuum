#pragma once

#include <utility>

#include <entt/entt.hpp>

namespace ic
{
    class Scene;

    class SceneEntity final
    {
    public:
        SceneEntity() = default;

        SceneEntity(entt::entity entity, Scene* scene)
            : m_entity(entity)
            , m_scene(scene)
        {
        }

        explicit operator bool() const
        {
            return m_entity != entt::null && m_scene != nullptr;
        }

        entt::entity handle() const
        {
            return m_entity;
        }

        template <typename Component, typename... Args>
        Component& addComponent(Args&&... args);

        template <typename Component, typename... Args>
        Component& addOrReplaceComponent(Args&&... args);

        template <typename Component>
        void removeComponent();

        template <typename Component>
        bool hasComponent() const;

        template <typename Component>
        Component& getComponent();

        template <typename Component>
        const Component& getComponent() const;

    private:
        entt::entity m_entity = entt::null;
        Scene* m_scene = nullptr;
    };
}

#include "ic/scene/scene_entity.inl"
