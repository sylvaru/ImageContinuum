#include "ic/common/ic_pch.h"
#include "ic/scene/scene.h"

#include "ic/scene/scene_entity.h"

namespace ic
{
    Scene::Scene() = default;

    Scene::Scene(std::string name)
        : m_name(std::move(name))
    {
    }

    Scene::~Scene()
    {
        clear();
    }

    SceneEntity Scene::createEntity(std::string_view name)
    {
        return createEntityWithId(allocateEntityId(), name);
    }

    SceneEntity Scene::createEntityWithId(uint64_t id, std::string_view name)
    {
        entt::entity entity = m_registry.create();

        m_registry.emplace<EntityIdComponent>(entity, id);
        m_registry.emplace<TransformComponent>(entity);
        m_registry.emplace<HierarchyComponent>(entity);
        m_registry.emplace<ActiveComponent>(entity, true);

        if (!name.empty())
        {
            m_registry.emplace<NameComponent>(entity, std::string(name));
        }

        m_entityById[id] = entity;
        m_nextEntityId = std::max(m_nextEntityId, id + 1);
        incrementVersion();

        return SceneEntity{ entity, this };
    }

    void Scene::destroyEntity(entt::entity entity)
    {
        if (!valid(entity))
        {
            return;
        }

        if (auto* id = m_registry.try_get<EntityIdComponent>(entity))
        {
            m_entityById.erase(id->id);
        }

        m_registry.destroy(entity);
        incrementVersion();
    }

    bool Scene::valid(entt::entity entity) const
    {
        return m_registry.valid(entity);
    }

    entt::entity Scene::findById(uint64_t id) const
    {
        auto it = m_entityById.find(id);
        return it != m_entityById.end() ? it->second : entt::null;
    }

    void Scene::clear()
    {
        m_registry.clear();
        m_entityById.clear();
        ++m_version;
    }

    uint64_t Scene::allocateEntityId()
    {
        return m_nextEntityId++;
    }
}
