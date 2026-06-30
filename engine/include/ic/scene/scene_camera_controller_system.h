#pragma once

#include <entt/entt.hpp>

namespace ic
{
    struct Input;

    class SceneCameraControllerSystem final
    {
    public:
        static void update(
            entt::registry& registry,
            Input& input,
            float deltaTime);
    };
}
