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
	{
	}
    AppBase::~AppBase() = default;

    int AppBase::run(int argc, char** argv)
    {
        (void)argc, argv;
        // create and initialize glfw platform context
        m_platform = std::make_unique<PlatformState>();

        // create window using spec
        m_window = std::make_unique<GLFWWindow>(m_spec.window);

        // create input bound to window
        m_input = std::make_unique<GLFWInput>(*m_window);

        m_window->bindEventSink([this](Event e)
            {
                m_eventQueue.push(e);
            });

        // create renderer after window exists
        //m_renderer = createRenderer(
        //    *m_window,
        //    m_spec.renderer
        //);

        // build services view
        m_services = AppServices{
            m_input.get(),
            m_window.get()
        };

        // user init
        onInit();

        // main loop
        while (!m_window->shouldClose())
        {
            m_input->beginFrame();

            m_window->pollEvents();

            m_eventQueue.drain([&](Event& e) {
                routeEvent(e);
                });
            
            onUpdate(1.0f / 60.0f);

            m_layerStack.updateAll(1.0f / 60.0f);
            m_layerStack.renderAll(1.0f);
        }

        onShutdown();
        return 0;
    }

    void AppBase::routeEvent(Event& e)
    {
        m_input->onEvent(e);
        m_layerStack.dispatchEvent(e);

        handleSystemEvents(e);
    }

    void AppBase::handleSystemEvents(const Event& e)
    {
        switch (e.type)
        {
        case EventType::KeyPressed:
        {
            if (e.key.key == IcKey::ESCAPE)
            {
                m_window->requestClose();
            }
            break;
        }

        case EventType::MouseButtonPressed:
        {
            spdlog::info("Mouse button: {}", (int)e.mouseButton.button);
            break;
        }

        default:
            break;
        }
    }
}