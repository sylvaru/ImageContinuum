// ic/platform/glfw_input.cpp
#include "ic/common/ic_pch.h"
#include "ic/platform/glfw_input.h"
#include "ic/platform/glfw_window.h"
#include "ic/core/events.h"


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

		m_keysPressed.reset();
		m_keysReleased.reset();

		for (int i = 0; i < 8; i++)
		{
			m_mousePressed[i] = false;
			m_mouseReleased[i] = false;
		}
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
			onMouseMove(e.mouseMove.x, e.mouseMove.y);
			break;

		case EventType::MouseButtonPressed:
			onMouseButtonPressed(e.mouseButton.button);
			break;

		case EventType::MouseButtonReleased:
			onMouseButtonReleased(e.mouseButton.button);
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
		size_t k = static_cast<size_t>(key);

		m_keys.set(k);
		m_keysPressed.set(k);
	}

	void GLFWInput::onKeyReleased(IcKey key)
	{
		size_t k = static_cast<size_t>(key);

		m_keys.reset(k);
		m_keysReleased.set(k);
	}

	void GLFWInput::onMouseMove(double x, double y)
	{
		if (!m_mouseInitialized)
		{
			m_mouseX = x;
			m_mouseY = y;
			m_mouseInitialized = true;
			return;
		}

		m_mouseDX += (x - m_mouseX);
		m_mouseDY += (y - m_mouseY);

		m_mouseX = x;
		m_mouseY = y;
	}

	void GLFWInput::onMouseButtonPressed(MouseButton b)
	{
		int i = static_cast<int>(b);

		m_mouseButtons[i] = true;
		m_mousePressed[i] = true;
	}

	void GLFWInput::onMouseButtonReleased(MouseButton b)
	{
		int i = static_cast<int>(b);

		m_mouseButtons[i] = false;
		m_mouseReleased[i] = true;
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

	bool GLFWInput::isKeyPressed(IcKey key) const
	{
		return m_keys.test(static_cast<size_t>(key));
	}

	bool GLFWInput::isMouseButtonPressed(MouseButton button) const
	{
		return m_mouseButtons[static_cast<int>(button)];
	}
}
