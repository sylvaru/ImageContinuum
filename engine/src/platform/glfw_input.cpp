// platform/glfw_input.cpp
#include "image_continuum/common/ic_pch.h"
#include "image_continuum/platform/glfw_input.h"
#include "image_continuum/platform/glfw_window.h"
#include "image_continuum/interface/events.h"


namespace ic
{
	GLFWInput::GLFWInput(Window& window)
		: m_window(window)
	{
	}

	void GLFWInput::beginFrame()
	{
		m_mouseDX = 0.0;
		m_mouseDY = 0.0;
		m_scrollX = 0.0;
		m_scrollY = 0.0;
	}


	void GLFWInput::onMouseMove(double x, double y)
	{
		if (!m_mouseInitialized)
		{
			m_lastMouseX = x;
			m_lastMouseY = y;
			m_mouseInitialized = true;
			return;
		}

		m_mouseDX += (x - m_lastMouseX);
		m_mouseDY += (y - m_lastMouseY);

		m_lastMouseX = x;
		m_lastMouseY = y;
	}

	bool GLFWInput::isKeyPressed(IcKey key) const
	{
		(void)key;
		return false;
	}

	bool GLFWInput::isMouseButtonPressed(MouseButton button) const
	{
		int glfwButton = 0;

		switch (button)
		{
		case MouseButton::Left:   glfwButton = GLFW_MOUSE_BUTTON_LEFT; break;
		case MouseButton::Right:  glfwButton = GLFW_MOUSE_BUTTON_RIGHT; break;
		case MouseButton::Middle: glfwButton = GLFW_MOUSE_BUTTON_MIDDLE; break;
		default: return false;
		}

		auto* window = static_cast<GLFWwindow*>(m_window.getNativeHandle());

		return glfwGetMouseButton(window, glfwButton) == GLFW_PRESS;
	}

	void GLFWInput::onEvent(const Event& e)
	{
		switch (e.type)
		{
		case EventType::KeyPressed:
			onKeyPressed(e.key.key);
			break;

		case EventType::KeyReleased:
			onKeyReleased(e.key.key);
			break;

		case EventType::MouseMoved:
			onMouseMove(
				e.mouseMove.x,
				e.mouseMove.y);
			break;

		case EventType::MouseButtonPressed:
			onMouseButtonPressed(
				e.mouseButton.button);
			break;

		case EventType::MouseButtonReleased:
			onMouseButtonReleased(
				e.mouseButton.button);
			break;

		case EventType::MouseScrolled:
			m_scrollX += e.scroll.dx;
			m_scrollY += e.scroll.dy;
			break;

		default:
			break;
		}
	}

	void GLFWInput::onKeyPressed(IcKey key)
	{
		m_keys.set(static_cast<size_t>(key));
	}

	void GLFWInput::onKeyReleased(IcKey key)
	{
		m_keys.reset(static_cast<size_t>(key));
	}

	void GLFWInput::consumeMouseDelta(double& dx, double& dy)
	{
		dx = m_mouseDX;
		dy = m_mouseDY;

		m_mouseDX = 0.0;
		m_mouseDY = 0.0;
	}

	void GLFWInput::consumeMouseScroll(double& dx, double& dy)
	{
		dx = m_scrollX;
		dy = m_scrollY;
		m_scrollX = 0.0;
		m_scrollY = 0.0;
	}

	void GLFWInput::lockCursor(bool lock)
	{
		m_cursorLocked = lock;

		auto* window = static_cast<GLFWwindow*>(m_window.getNativeHandle());

		glfwSetInputMode(
			window,
			GLFW_CURSOR,
			lock ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL
		);
	}

	bool GLFWInput::isCursorLocked() const
	{
		return m_cursorLocked;
	}

	void GLFWInput::onMouseButtonPressed(MouseButton b)
	{
		(void)b;
	}

	void GLFWInput::onMouseButtonReleased(MouseButton b)
	{
		(void)b;
	}
}