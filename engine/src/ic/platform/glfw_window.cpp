// ic/platform/glfw_window.cpp
#include "ic/common/ic_pch.h"
#include "ic/platform/glfw_window.h"
#include "ic/core/events.h"

namespace ic
{
    GLFWWindow::GLFWWindow(const WindowSpecification& spec)
    {
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        m_window = glfwCreateWindow(
            spec.width,
            spec.height,
            spec.title.c_str(),
            nullptr,
            nullptr
        );

        if (!m_window)
            throw std::runtime_error("Failed to create GLFW window");

        glfwSetWindowUserPointer(m_window, this);

        int framebufferWidth = 0;
        int framebufferHeight = 0;
        glfwGetFramebufferSize(
            m_window,
            &framebufferWidth,
            &framebufferHeight);
        m_width = static_cast<uint32_t>(std::max(framebufferWidth, 0));
        m_height = static_cast<uint32_t>(std::max(framebufferHeight, 0));

        initCallbacks();
    }

    GLFWWindow::~GLFWWindow()
    {
        if (m_window)
            glfwDestroyWindow(m_window);
    }

    void GLFWWindow::requestClose()
    {
        glfwSetWindowShouldClose(m_window, GLFW_TRUE);
    }

    void GLFWWindow::pollEvents()
    {
        glfwPollEvents();
    }

    bool GLFWWindow::shouldClose() const
    {
        return glfwWindowShouldClose(m_window);
    }

    uint32_t GLFWWindow::getWidth() const
    {
        return m_width;
    }

    uint32_t GLFWWindow::getHeight() const
    {
        return m_height;
    }

    void* GLFWWindow::getNativeHandle() const
    {
        return m_window;
    }

    void GLFWWindow::setTitle(std::string_view title)
    {
        glfwSetWindowTitle(
            m_window,
            std::string(title).c_str());
    }

    void GLFWWindow::bindEventSink(EventCallbackFn fn)
    {
        m_eventCallback = std::move(fn);
    }

    void GLFWWindow::initCallbacks()
    {
        setupKeyCallback();
        setupMouseCallback();
        setupCursorCallback();
        setupScrollCallback();
        setupFramebufferSizeCallback();
    }

    void GLFWWindow::setupKeyCallback()
    {
        glfwSetKeyCallback(m_window,
            [](GLFWwindow* w, int key, int, int action, int)
            {
                auto* self = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(w));
                if (!self || !self->m_eventCallback) return;

                switch (action)
                {
                case GLFW_PRESS:
                    self->m_eventCallback(makeKeyPressed((IcKey)key));
                    break;

                case GLFW_RELEASE:
                    self->m_eventCallback(makeKeyReleased((IcKey)key));
                    break;

                default:
                    break;
                }
            });
    }

    void GLFWWindow::setupMouseCallback()
    {
        glfwSetMouseButtonCallback(m_window,
            [](GLFWwindow* w, int button, int action, int)
            {
                auto* self = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(w));
                if (!self || !self->m_eventCallback) return;

                switch (action)
                {
                case GLFW_PRESS:
                    self->m_eventCallback(makeMouseButtonPressed((MouseButton)button));
                    break;

                case GLFW_RELEASE:
                    self->m_eventCallback(makeMouseButtonReleased((MouseButton)button));
                    break;

                default:
                    break;
                }
            });
    }

    void GLFWWindow::setupCursorCallback()
    {
        glfwSetCursorPosCallback(m_window,
            [](GLFWwindow* w, double x, double y)
            {
                auto* self = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(w));
                if (!self || !self->m_eventCallback) return;

                self->m_eventCallback(makeMouseMoved(x, y));
            });
    }

    void GLFWWindow::setupScrollCallback()
    {
        glfwSetScrollCallback(m_window,
            [](GLFWwindow* w, double dx, double dy)
            {
                auto* self = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(w));
                if (!self || !self->m_eventCallback) return;

                self->m_eventCallback(makeMouseScrolled(dx, dy));
            });
    }

    void GLFWWindow::setupFramebufferSizeCallback()
    {
        glfwSetFramebufferSizeCallback(m_window,
            [](GLFWwindow* w, int width, int height)
            {
                auto* self = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(w));
                if (!self) return;

                self->m_width = static_cast<uint32_t>(std::max(width, 0));
                self->m_height = static_cast<uint32_t>(std::max(height, 0));

                if (self->m_eventCallback)
                {
                    self->m_eventCallback(
                        makeWindowResize(self->m_width, self->m_height));
                }
            });
    }
}
