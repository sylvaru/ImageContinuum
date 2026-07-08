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

		const bool focused = isWindowFocused();
		if (focused != m_windowFocused)
		{
			m_windowFocused = focused;
			if (!m_windowFocused && m_cursorLocked)
			{
				lockCursor(false);
			}

			syncMousePosition();
		}
	}

	void GLFWInput::onEvent(const Event& e)
	{
		switch (e.type)
		{
		case EventType::KeyPressed:
		{
			const KeyEvent* key = getPayload<KeyEvent>(e);
			assert(key);

			onKeyPressed(key->key);
			break;
		}

		case EventType::KeyReleased:
		{
			const KeyEvent* key = getPayload<KeyEvent>(e);
			assert(key);

			onKeyReleased(key->key);
			break;
		}

		case EventType::MouseMoved:
		{
			const MouseMoveEvent* mouseMove = getPayload<MouseMoveEvent>(e);
			assert(mouseMove);

			onMouseMove(mouseMove->x, mouseMove->y);
			break;
		}

		case EventType::MouseButtonPressed:
		{
			const MouseButtonEvent* mouseButton = getPayload<MouseButtonEvent>(e);
			assert(mouseButton);

			onMouseButtonPressed(mouseButton->button);
			break;
		}

		case EventType::MouseButtonReleased:
		{
			const MouseButtonEvent* mouseButton = getPayload<MouseButtonEvent>(e);
			assert(mouseButton);

			onMouseButtonReleased(mouseButton->button);
			break;
		}

		case EventType::MouseScrolled:
		{
			const ScrollEvent* scroll = getPayload<ScrollEvent>(e);
			assert(scroll);

			m_scrollX += scroll->dx;
			m_scrollY += scroll->dy;
			break;
		}

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
		const bool focused = isWindowFocused();
		if (!focused)
		{
			m_windowFocused = false;
			m_mouseX = x;
			m_mouseY = y;
			m_mouseInitialized = true;
			m_mouseDX = 0.0;
			m_mouseDY = 0.0;
			return;
		}

		if (!m_windowFocused || !m_mouseInitialized)
		{
			m_windowFocused = true;
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
		auto* window = static_cast<GLFWwindow*>(m_window.getNativeHandle());
		if (lock && glfwGetWindowAttrib(window, GLFW_FOCUSED) != GLFW_TRUE)
		{
			lock = false;
		}

		m_cursorLocked = lock;

		glfwSetInputMode(
			window,
			GLFW_CURSOR,
			lock ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL
		);

		syncMousePosition();
	}

	bool GLFWInput::isCursorLocked() const
	{
		return m_cursorLocked;
	}

	bool GLFWInput::isKeyPressed(IcKey key) const
	{
		return m_keys.test(static_cast<size_t>(key));
	}

	bool GLFWInput::wasKeyPressed(IcKey key) const
	{
		return m_keysPressed.test(static_cast<size_t>(key));
	}

	bool GLFWInput::isMouseButtonPressed(MouseButton button) const
	{
		return m_mouseButtons[static_cast<int>(button)];
	}

	void GLFWInput::syncMousePosition()
	{
		auto* window = static_cast<GLFWwindow*>(m_window.getNativeHandle());

		double x = 0.0;
		double y = 0.0;
		glfwGetCursorPos(window, &x, &y);

		m_mouseX = x;
		m_mouseY = y;
		m_mouseInitialized = true;
		m_mouseDX = 0.0;
		m_mouseDY = 0.0;
	}

	bool GLFWInput::isWindowFocused() const
	{
		auto* window = static_cast<GLFWwindow*>(m_window.getNativeHandle());
		return glfwGetWindowAttrib(window, GLFW_FOCUSED) == GLFW_TRUE;
	}
}
