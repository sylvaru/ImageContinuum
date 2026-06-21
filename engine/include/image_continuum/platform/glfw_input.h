// image_continuum/platform/glfw_input.h
#pragma once
#include "image_continuum/interface/input.h"
#include <bitset>
#include <string>
#include <GLFW/glfw3.h>

namespace ic
{
    class Window;

    class GLFWInput final : public Input
    {
    public:
        explicit GLFWInput(Window& window);

        void beginFrame() override;

        void onKeyPressed(IcKey key) override;
        void onKeyReleased(IcKey key) override;

        void onMouseMove(double x, double y) override;
        void onMouseButtonPressed(MouseButton b) override;
        void onMouseButtonReleased(MouseButton b) override;

        bool isKeyPressed(IcKey key) const override;
        bool isMouseButtonPressed(MouseButton button) const override;

        void consumeMouseDelta(double& dx, double& dy) override;
        void consumeMouseScroll(double& dx, double& dy) override;

        void lockCursor(bool lock) override;
        bool isCursorLocked() const override;
        
        void onEvent(const Event& e) override;
    private:
        Window& m_window;

        std::bitset<512> m_keys;

        bool m_mouseInitialized = false;
        double m_lastMouseX = 0.0;
        double m_lastMouseY = 0.0;

        double m_mouseDX = 0.0;
        double m_mouseDY = 0.0;

        double m_scrollX = 0.0;
        double m_scrollY = 0.0;

        bool m_cursorLocked = true;
    };
}