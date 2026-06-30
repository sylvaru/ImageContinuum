// ic/platform/glfw_window.h
#pragma once
#include <GLFW/glfw3.h>
#include "ic/interface/window.h"

namespace ic
{
    class GLFWWindow final : public Window
    {
    public:
        explicit GLFWWindow(
            const WindowSpecification& spec);

        ~GLFWWindow() override;

        void pollEvents() override;

        bool shouldClose() const override;

        uint32_t getWidth() const override;
        uint32_t getHeight() const override;

        void* getNativeHandle() const override;

        void setTitle(std::string_view title) override;

        void requestClose() override;

        void bindEventSink(EventCallbackFn fn) override;

        void initCallbacks();

        void setupKeyCallback();

        void setupMouseCallback();
        void setupCursorCallback();
        void setupScrollCallback();

  
    private:
        GLFWwindow* m_window = nullptr;
        uint32_t m_width = 0;
        uint32_t m_height = 0;

        EventCallbackFn m_eventCallback;
    };


}
