#pragma once

#include <cstdint>
#include <filesystem>

#include <entt/entt.hpp>

#include "ic/renderer/renderer_specification.h"

namespace ic
{
    using SceneEntityHandle = entt::entity;

    static constexpr SceneEntityHandle kNullEntity = entt::null;

    struct SceneHandle
    {
        uint32_t index = UINT32_MAX;
        uint32_t generation = 0;

        explicit operator bool() const
        {
            return index != UINT32_MAX;
        }

        friend bool operator==(SceneHandle a, SceneHandle b)
        {
            return a.index == b.index && a.generation == b.generation;
        }

        friend bool operator!=(SceneHandle a, SceneHandle b)
        {
            return !(a == b);
        }
    };

    static constexpr SceneHandle kInvalidSceneHandle{};

    enum class SceneState : uint8_t
    {
        Empty = 0,
        Loading,
        Loaded,
        Failed
    };

    enum class SceneLoadMode : uint8_t
    {
        ReplaceActive = 0,
        Additive
    };

    struct SceneManagerDesc
    {
        bool enableAsyncSceneLoading = true;
        uint32_t maxQueuedSceneLoads = 4;
        std::filesystem::path modelRoot;
        EnvironmentSettings defaultEnvironment;
    };

    struct SceneLoadOptions
    {
        SceneLoadMode mode = SceneLoadMode::ReplaceActive;
        bool loadReferencedAssets = true;
        bool makeActive = true;
    };
}
