// core/app_base.cpp
#include "image_continuum/common/ic_pch.h"
#include "image_continuum/core/app_base.h"
#include "image_continuum/platform/glfw_window.h"
#include "image_continuum/platform/glfw_input.h"
#include "image_continuum/interface/window.h"
#include "image_continuum/platform/glfw_context.h"
#include "image_continuum/interface/events.h"
#include "image_continuum/interface/input.h"

#include <spdlog/spdlog.h>

namespace ic
{
    struct AppBase::PlatformState
    {
        GLFWContext glfw;
    };

	AppBase::AppBase(AppSpecification spec)
		: m_spec(std::move(spec))
        , m_executor(*this)
	{
	}
    AppBase::~AppBase() = default;

    void AppBase::initializeAppBase()
    {
        createPlatform();
        createWindow();
        createInput();

        bindEvents();

        buildServices();

        onInit();
    }

    int AppBase::run([[maybe_unused]] int argc, [[maybe_unused]] char** argv)
    {
        
        initializeAppBase();

        // main loop
        while (!m_window->shouldClose())
        {
            m_input->beginFrame();

            m_window->pollEvents();

            m_eventFrameBuffer.beginFrame(m_frame);

            m_executor.execute(m_frame);
        }

        shutdown();
        return 0;
    }

    void AppBase::dispatchEvent(EventChannel channel, Event& e)
    {
        // global systems first
        m_input->onEvent(e);

        // channel based routing
        switch (channel)
        {
        case EventChannel::Input:
            handleInputEvent(e);
            break;

        case EventChannel::Window:
            handleWindowEvent(e);
            break;

        case EventChannel::Render:
            handleRenderEvent(e);
            break;
        }

        // always reach layers
        m_layerStack.dispatchEvent(e);
    }

    void AppBase::handleInputEvent(Event& e)
    {
        m_input->onEvent(e);

        if (e.type == EventType::KeyPressed &&
            e.key.key == IcKey::ESCAPE)
        {
            m_window->requestClose();
        }

        if (e.type == EventType::MouseButtonPressed)
        {
            spdlog::info(
                "Mouse button: {}",
                (int)e.mouseButton.button);
        }
    }
    void AppBase::handleWindowEvent(Event& e)
    {
        switch (e.type)
        {
        case EventType::WindowResize:
            // renderer resize later
            break;

        case EventType::WindowClose:
            m_window->requestClose();
            break;

        default:
            break;
        }
    }

    void AppBase::handleRenderEvent(Event&)
    {
        // reserved for render driven events later
    }

    void AppBase::createPlatform()
    {
        m_platform =
            std::make_unique<PlatformState>();
    }

    void AppBase::createWindow()
    {
        m_window =
            std::make_unique<GLFWWindow>(
                m_spec.window);
    }

    void AppBase::createInput()
    {
        m_input =
            std::make_unique<GLFWInput>(
                *m_window);
    }

    void AppBase::bindEvents()
    {
        m_window->bindEventSink(
            [this](const Event& e)
            {
                m_eventFrameBuffer.push(
                    channelForEvent(e.type),
                    e);
            });
    }

    void AppBase::buildServices()
    {
        m_services =
        {
            m_input.get(),
            m_window.get()
        };
    }

    void AppBase::shutdown()
    {
        onShutdown();
    }
}