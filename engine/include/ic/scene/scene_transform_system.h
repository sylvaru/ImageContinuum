#pragma once

#include <entt/entt.hpp>
#include <glm/glm.hpp>

namespace ic
{
    class SceneTransformSystem final
    {
    public:
        static void update(entt::registry& registry);

    private:
        static void updateEntityRecursive(
            entt::registry& registry,
            entt::entity entity,
            const glm::mat4& parentWorld,
            bool parentDirty);
    };
}
