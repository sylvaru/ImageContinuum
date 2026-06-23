// ic/interface/input.h
#pragma once
#include "ic/common/ic_key_codes.h"

namespace ic
{
    struct Event;

    struct Input
    {
        virtual ~Input() = default;

        virtual void beginFrame() = 0;

        virtual void onKeyPressed(IcKey key) = 0;
        virtual void onKeyReleased(IcKey key) = 0;

        virtual void onMouseMove(double x, double y) = 0;
        virtual void onMouseButtonPressed(MouseButton b) = 0;
        virtual void onMouseButtonReleased(MouseButton b) = 0;

        virtual bool isKeyPressed(IcKey key) const = 0;
        virtual bool isMouseButtonPressed(MouseButton button) const = 0;

        virtual void consumeMouseDelta(double& dx, double& dy) = 0;
        virtual void consumeMouseScroll(double& dx, double& dy) = 0;

        virtual void lockCursor(bool lock) = 0;
        virtual bool isCursorLocked() const = 0;

        virtual void onEvent(const Event& e) = 0;
    };
}