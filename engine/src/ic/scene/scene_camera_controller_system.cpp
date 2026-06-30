#include "ic/common/ic_pch.h"
#include "ic/scene/scene_camera_controller_system.h"

#include <algorithm>

#include <glm/gtc/constants.hpp>

#include "ic/common/ic_key_codes.h"
#include "ic/interface/input.h"
#include "ic/scene/scene_components.h"
#include "ic/scene/scene_transform_utils.h"

namespace ic
{
    namespace
    {
        bool keyDown(const Input& input, IcKey key)
        {
            return input.isKeyPressed(key);
        }
    }

    void SceneCameraControllerSystem::update(
        entt::registry& registry,
        Input& input,
        float deltaTime)
    {
        double mouseDx = 0.0;
        double mouseDy = 0.0;
        input.consumeMouseDelta(mouseDx, mouseDy);

        auto view =
            registry.view<TransformComponent, CameraComponent, FlyCameraControllerComponent, ActiveComponent>();

        for (entt::entity entity : view)
        {
            const auto& active = view.get<ActiveComponent>(entity);
            if (!active.active)
            {
                continue;
            }

            auto& transform = view.get<TransformComponent>(entity);
            auto& controller = view.get<FlyCameraControllerComponent>(entity);

            if (input.wasKeyPressed(IcKey::LEFT_ALT))
            {
                controller.captureMouse = !controller.captureMouse;
                if (!controller.captureMouse && input.isCursorLocked())
                {
                    input.lockCursor(false);
                }
            }
            else if (!controller.captureMouse &&
                (input.isMouseButtonPressed(MouseButton::Right) ||
                 input.isMouseButtonPressed(MouseButton::Left)))
            {
                controller.captureMouse = true;
            }

            if (controller.captureMouse && !input.isCursorLocked())
            {
                input.lockCursor(true);
            }

            bool changed = false;

            if (controller.captureMouse)
            {
                controller.yaw -= static_cast<float>(mouseDx) *
                    controller.mouseSensitivity;
                controller.pitch -= static_cast<float>(mouseDy) *
                    controller.mouseSensitivity;

                constexpr float maxPitch = glm::half_pi<float>() - 0.01f;
                controller.pitch = std::clamp(
                    controller.pitch,
                    -maxPitch,
                    maxPitch);

                const glm::quat yawRotation =
                    glm::angleAxis(controller.yaw, glm::vec3(0.0f, 1.0f, 0.0f));
                const glm::quat pitchRotation =
                    glm::angleAxis(controller.pitch, glm::vec3(1.0f, 0.0f, 0.0f));

                transform.rotation = glm::normalize(yawRotation * pitchRotation);
                changed = mouseDx != 0.0 || mouseDy != 0.0;
            }

            glm::vec3 movement(0.0f);
            if (keyDown(input, IcKey::W))
            {
                movement += transform.rotation * glm::vec3(0.0f, 0.0f, -1.0f);
            }
            if (keyDown(input, IcKey::S))
            {
                movement += transform.rotation * glm::vec3(0.0f, 0.0f, 1.0f);
            }
            if (keyDown(input, IcKey::D))
            {
                movement += transform.rotation * glm::vec3(1.0f, 0.0f, 0.0f);
            }
            if (keyDown(input, IcKey::A))
            {
                movement += transform.rotation * glm::vec3(-1.0f, 0.0f, 0.0f);
            }
            if (keyDown(input, IcKey::E))
            {
                movement += glm::vec3(0.0f, 1.0f, 0.0f);
            }
            if (keyDown(input, IcKey::Q))
            {
                movement += glm::vec3(0.0f, -1.0f, 0.0f);
            }

            if (glm::dot(movement, movement) > 0.0f)
            {
                movement = glm::normalize(movement);

                float speed = controller.moveSpeed;
                if (keyDown(input, IcKey::LEFT_SHIFT) ||
                    keyDown(input, IcKey::RIGHT_SHIFT))
                {
                    speed *= controller.fastMoveMultiplier;
                }

                transform.translation += movement * speed * deltaTime;
                changed = true;
            }

            if (changed)
            {
                markTransformDirty(registry, entity);
            }
        }
    }
}
