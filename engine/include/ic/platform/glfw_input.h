// ic/platform/glfw_input.h
#pragma once
#include "ic/interface/input.h"
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

        // Event ingestion
        void onEvent(const Event& e) override;

        // Query API (frame snapshot)
        bool isKeyPressed(IcKey key) const override;
        bool isMouseButtonPressed(MouseButton button) const override;

        void consumeMouseDelta(double& dx, double& dy) override;
        void consumeMouseScroll(double& dx, double& dy) override;

        void lockCursor(bool lock) override;
        bool isCursorLocked() const override;



        void onKeyPressed(IcKey key) override;
        void onKeyReleased(IcKey key) override;

        void onMouseMove(double x, double y) override;
        void onMouseButtonPressed(MouseButton b) override;
        void onMouseButtonReleased(MouseButton b) override;
    private:
        Window& m_window;

        // Key state
        std::bitset<512> m_keys;
        std::bitset<512> m_keysPressed;
        std::bitset<512> m_keysReleased;

        // Mouse state
        bool m_mouseInitialized = false;

        double m_mouseX = 0.0;
        double m_mouseY = 0.0;

        double m_mouseDX = 0.0;
        double m_mouseDY = 0.0;

        double m_scrollX = 0.0;
        double m_scrollY = 0.0;

        bool m_cursorLocked = false;

        // Mouse buttons
        bool m_mouseButtons[8]{};
        bool m_mousePressed[8]{};
        bool m_mouseReleased[8]{};
    };
}
