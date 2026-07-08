// ic/interface/events.h
#pragma once

#include "ic/common/ic_key_codes.h"

#include <cassert>
#include <cstdint>
#include <variant>

namespace ic
{
    enum class EventType : uint16_t
    {
        None = 0,

        KeyPressed,
        KeyReleased,

        MouseMoved,
        MouseButtonPressed,
        MouseButtonReleased,
        MouseScrolled,

        WindowResize,
        WindowClose,

        ShaderReloaded,
        SwapchainRecreated,

        AssetLoaded,
        AssetUnloaded,

        PhysicsBodyCreated,
        PhysicsBodyDestroyed
    };

    enum class EventChannel
    {
        Input,
        Window,

        Asset,
        Physics,
        Renderer,

        System,

        Count
    };

    constexpr size_t kEventChannelCount =
        static_cast<size_t>(EventChannel::Count);

    struct KeyEvent
    {
        IcKey key{};
    };

    struct MouseButtonEvent
    {
        MouseButton button{};
    };

    struct MouseMoveEvent
    {
        double x = 0.0;
        double y = 0.0;
    };

    struct ScrollEvent
    {
        double dx = 0.0;
        double dy = 0.0;
    };

    struct ResizeEvent
    {
        uint32_t width = 0;
        uint32_t height = 0;
    };

    using EventPayload = std::variant<
        std::monostate,
        KeyEvent,
        MouseMoveEvent,
        MouseButtonEvent,
        ScrollEvent,
        ResizeEvent>;

    struct Event
    {
        EventType type = EventType::None;
        EventPayload payload = std::monostate{};
    };

    [[nodiscard]] constexpr EventChannel channelForEvent(EventType type)
    {
        switch (type)
        {
        case EventType::KeyPressed:
        case EventType::KeyReleased:
        case EventType::MouseMoved:
        case EventType::MouseButtonPressed:
        case EventType::MouseButtonReleased:
        case EventType::MouseScrolled:
            return EventChannel::Input;

        case EventType::WindowResize:
        case EventType::WindowClose:
            return EventChannel::Window;

        case EventType::ShaderReloaded:
        case EventType::SwapchainRecreated:
            return EventChannel::Renderer;

        case EventType::AssetLoaded:
        case EventType::AssetUnloaded:
            return EventChannel::Asset;

        case EventType::PhysicsBodyCreated:
        case EventType::PhysicsBodyDestroyed:
            return EventChannel::Physics;

        case EventType::None:
        default:
            return EventChannel::System;
        }
    }

    [[nodiscard]] constexpr bool isInputEvent(EventType type)
    {
        return channelForEvent(type) == EventChannel::Input;
    }

    [[nodiscard]] constexpr bool isWindowEvent(EventType type)
    {
        return channelForEvent(type) == EventChannel::Window;
    }

    template <typename T>
    [[nodiscard]] constexpr bool hasPayload(const Event& event)
    {
        return std::holds_alternative<T>(event.payload);
    }

    template <typename T>
    [[nodiscard]] constexpr const T* getPayload(const Event& event)
    {
        return std::get_if<T>(&event.payload);
    }

    template <typename T>
    [[nodiscard]] constexpr T* getPayload(Event& event)
    {
        return std::get_if<T>(&event.payload);
    }

    [[nodiscard]] constexpr Event makeKeyPressed(IcKey key)
    {
        return Event{
            .type = EventType::KeyPressed,
            .payload = KeyEvent{
                .key = key
            }
        };
    }

    [[nodiscard]] constexpr Event makeKeyReleased(IcKey key)
    {
        return Event{
            .type = EventType::KeyReleased,
            .payload = KeyEvent{
                .key = key
            }
        };
    }

    [[nodiscard]] constexpr Event makeMouseMoved(
        double x,
        double y)
    {
        return Event{
            .type = EventType::MouseMoved,
            .payload = MouseMoveEvent{
                .x = x,
                .y = y
            }
        };
    }

    [[nodiscard]] constexpr Event makeMouseScrolled(
        double dx,
        double dy)
    {
        return Event{
            .type = EventType::MouseScrolled,
            .payload = ScrollEvent{
                .dx = dx,
                .dy = dy
            }
        };
    }

    [[nodiscard]] constexpr Event makeMouseButtonPressed(
        MouseButton button)
    {
        return Event{
            .type = EventType::MouseButtonPressed,
            .payload = MouseButtonEvent{
                .button = button
            }
        };
    }

    [[nodiscard]] constexpr Event makeMouseButtonReleased(
        MouseButton button)
    {
        return Event{
            .type = EventType::MouseButtonReleased,
            .payload = MouseButtonEvent{
                .button = button
            }
        };
    }

    [[nodiscard]] constexpr Event makeWindowClose()
    {
        return Event{
            .type = EventType::WindowClose,
            .payload = std::monostate{}
        };
    }

    [[nodiscard]] constexpr Event makeWindowResize(
        uint32_t width,
        uint32_t height)
    {
        return Event{
            .type = EventType::WindowResize,
            .payload = ResizeEvent{
                .width = width,
                .height = height
            }
        };
    }
}