#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>

#include <entt/entt.hpp>

#include "ic/core/asset_manager.h"
#include "ic/renderer/renderer_specification.h"
#include "ic/scene/scene_components.h"
#include "ic/scene/scene_types.h"

namespace ic
{
    class SceneEntity;

    class Scene final
    {
    public:
        Scene();
        explicit Scene(std::string name);
        ~Scene();

        Scene(const Scene&) = delete;
        Scene& operator=(const Scene&) = delete;

        Scene(Scene&&) noexcept = default;
        Scene& operator=(Scene&&) noexcept = default;

        entt::registry& registry()
        {
            return m_registry;
        }

        const entt::registry& registry() const
        {
            return m_registry;
        }

        const std::string& name() const
        {
            return m_name;
        }

        void setName(std::string name)
        {
            m_name = std::move(name);
        }

        const std::filesystem::path& sourcePath() const
        {
            return m_sourcePath;
        }

        void setSourcePath(std::filesystem::path path)
        {
            m_sourcePath = std::move(path);
        }

        uint64_t version() const
        {
            return m_version;
        }

        AssetHandle environmentTexture() const
        {
            return m_environmentTexture;
        }

        void setEnvironmentTexture(AssetHandle handle)
        {
            m_environmentTexture = handle;
            incrementEnvironmentVersion();
            incrementVersion();
        }

        const EnvironmentSettings& environmentSettings() const
        {
            return m_environmentSettings;
        }

        uint64_t environmentVersion() const
        {
            return m_environmentVersion;
        }

        float environmentIntensity() const
        {
            return m_environmentSettings.intensity;
        }

        void setEnvironmentIntensity(float intensity)
        {
            m_environmentSettings.intensity = intensity;
            incrementEnvironmentVersion();
        }

        void setEnvironmentSettings(const EnvironmentSettings& settings)
        {
            const bool resourceDirty =
                m_environmentSettings.enabled != settings.enabled ||
                m_environmentSettings.cubemapSize != settings.cubemapSize;
            m_environmentSettings = settings;
            incrementEnvironmentVersion();
            if (resourceDirty)
            {
                incrementVersion();
            }
        }

        void incrementEnvironmentVersion()
        {
            ++m_environmentVersion;
        }

        void incrementVersion()
        {
            ++m_version;
        }

        SceneEntity createEntity(std::string_view name = {});
        SceneEntity createEntityWithId(uint64_t id, std::string_view name = {});

        void destroyEntity(entt::entity entity);

        bool valid(entt::entity entity) const;
        entt::entity findById(uint64_t id) const;

        void clear();

    private:
        uint64_t allocateEntityId();

        entt::registry m_registry;

        std::string m_name = "Untitled Scene";
        std::filesystem::path m_sourcePath;
        AssetHandle m_environmentTexture = {};
        EnvironmentSettings m_environmentSettings = {};
        uint64_t m_environmentVersion = 1;

        uint64_t m_nextEntityId = 1;
        uint64_t m_version = 1;

        std::unordered_map<uint64_t, entt::entity> m_entityById;
    };
}
