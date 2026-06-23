// ic/interface/events.h
#pragma once
#include "ic/common/ic_key_codes.h"

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

    struct Event
    {
        EventType type = EventType::None;

        union
        {
            struct
            {
                IcKey key;
            } key;

            struct
            {
                MouseButton button;
            } mouseButton;

            struct
            {
                double x;
                double y;
            } mouseMove;

            struct
            {
                double dx;
                double dy;
            } scroll;

            struct
            {
                uint32_t width;
                uint32_t height;
            } resize;
        };

    };


    constexpr EventChannel
        channelForEvent(const EventType t)
    {
        switch (t)
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

        default:
            return EventChannel::System;
        }
    }

    constexpr bool isInputEvent(EventType t)
    {
        return channelForEvent(t) == EventChannel::Input;
    }

    constexpr bool isWindowEvent(EventType type)
    {
        switch (type)
        {
        case EventType::WindowResize:
        case EventType::WindowClose:
            return true;

        default:
            return false;
        }
    }

    [[nodiscard]] constexpr Event makeKeyPressed(IcKey key)
    {
        Event e{};
        e.type = EventType::KeyPressed;
        e.key.key = key;
        return e;
    }

    [[nodiscard]] constexpr Event makeKeyReleased(IcKey key)
    {
        Event e{};
        e.type = EventType::KeyReleased;
        e.key.key = key;
        return e;
    }

    [[nodiscard]] constexpr Event makeMouseMoved(
        double x,
        double y)
    {
        Event e{};
        e.type = EventType::MouseMoved;
        e.mouseMove.x = x;
        e.mouseMove.y = y;
        return e;
    }

    [[nodiscard]] constexpr Event makeMouseScrolled(
        double dx,
        double dy)
    {
        Event e{};
        e.type = EventType::MouseScrolled;
        e.scroll.dx = dx;
        e.scroll.dy = dy;
        return e;
    }

    [[nodiscard]] constexpr Event makeMouseButtonPressed(
        MouseButton button)
    {
        Event e{};
        e.type = EventType::MouseButtonPressed;
        e.mouseButton.button = button;
        return e;
    }

    [[nodiscard]] constexpr Event makeMouseButtonReleased(
        MouseButton button)
    {
        Event e{};
        e.type = EventType::MouseButtonReleased;
        e.mouseButton.button = button;
        return e;
    }

    [[nodiscard]] constexpr Event makeWindowClose()
    {
        Event e{};
        e.type = EventType::WindowClose;
        return e;
    }

    [[nodiscard]] constexpr Event makeWindowResize(
        uint32_t width,
        uint32_t height)
    {
        Event e{};
        e.type = EventType::WindowResize;
        e.resize.width = width;
        e.resize.height = height;
        return e;
    }
   
}


