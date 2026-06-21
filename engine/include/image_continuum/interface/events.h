// image_continuum/interface/events.h
#pragma once
#include "image_continuum/common/ic_key_codes.h"

namespace ic
{
    enum class EventType : uint8_t
    {
        None = 0,
        KeyPressed,
        KeyReleased,

        MouseMoved,
        MouseButtonPressed,
        MouseButtonReleased,
        MouseScrolled,

        WindowResize,
        WindowClose
    };

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

    constexpr Event makeKeyPressed(IcKey key)
    {
        Event e{};
        e.type = EventType::KeyPressed;
        e.key.key = key;
        return e;
    }

    constexpr Event makeKeyReleased(IcKey key)
    {
        Event e{};
        e.type = EventType::KeyReleased;
        e.key.key = key;
        return e;
    }

    constexpr Event makeMouseMoved(
        double x,
        double y)
    {
        Event e{};
        e.type = EventType::MouseMoved;
        e.mouseMove.x = x;
        e.mouseMove.y = y;
        return e;
    }

    constexpr Event makeMouseScrolled(
        double dx,
        double dy)
    {
        Event e{};
        e.type = EventType::MouseScrolled;
        e.scroll.dx = dx;
        e.scroll.dy = dy;
        return e;
    }

    constexpr Event makeMouseButtonPressed(
        MouseButton button)
    {
        Event e{};
        e.type = EventType::MouseButtonPressed;
        e.mouseButton.button = button;
        return e;
    }

    constexpr Event makeMouseButtonReleased(
        MouseButton button)
    {
        Event e{};
        e.type = EventType::MouseButtonReleased;
        e.mouseButton.button = button;
        return e;
    }

    constexpr Event makeWindowClose()
    {
        Event e{};
        e.type = EventType::WindowClose;
        return e;
    }

    constexpr Event makeWindowResize(
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


