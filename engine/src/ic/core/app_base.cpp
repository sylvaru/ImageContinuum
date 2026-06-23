// core/app_base.cpp
#include "ic/common/ic_pch.h"
#include "ic/core/app_base.h"
#include "ic/platform/glfw_window.h"
#include "ic/platform/glfw_input.h"
#include "ic/interface/window.h"
#include "ic/platform/glfw_context.h"
#include "ic/core/events.h"
#include "ic/interface/input.h"
#include "ic/core/job_system.h"
#include "ic/util/profiler.h"
#include "ic/core/event_frame_buffer.h"
#include "ic/core/frame_executor.h"
#include <spdlog/spdlog.h>


namespace ic
{
    struct AppBase::PlatformState
    {
        GLFWContext glfw;
    };

    AppBase::AppBase(AppSpecification spec)
        : m_spec(std::move(spec))
        , m_runtime(std::make_unique<AppRuntime>(*this))
	{
	}
    
    AppBase::~AppBase() = default;

    struct AppBase::AppRuntime
    {
        Scope<PlatformState> platform;
        Scope<Window> window;
        Scope<Input> input;
        Scope<JobSystem> jobs;
        Scope<EventQueue> eventQueue;

        EventFrameBuffer eventBuffer;
        FrameExecutor executor;

        AppRuntime(AppBase& app)
            : executor(app)
            , eventQueue(std::make_unique<EventQueue>())
        {
        }
    };

    void AppBase::initAppBase(
        [[maybe_unused]] int argc,
        [[maybe_unused]] char** argv)
    {
        // Todo: use toml++ for easily configuring
        // static initialization data

        createPlatform();
        createWindow();
        createInput();
        createJobSystem();

        bindEvents();
        buildServices();

    }

    int AppBase::run(
        [[maybe_unused]] int argc,
        [[maybe_unused]] char** argv)
    {
        
        initAppBase(argc, argv);
        onInit(); // Client app configuration happens here

        m_clock.reset();
        m_frame.jobs = m_runtime->jobs.get();
        m_frame.input = m_runtime->input.get();

        // Main Loop
        while (!m_runtime->window->shouldClose())
        {
            //ZoneScopedN("Frame");

            auto frameStart = std::chrono::steady_clock::now();

            m_runtime->window->pollEvents();
            m_runtime->input->beginFrame();

            m_frame.deltaTime = m_clock.tick();
            m_frame.timeSinceStart = m_clock.getTimeSinceStart();

            m_runtime->eventBuffer.ingest(*m_runtime->eventQueue);
            m_runtime->eventBuffer.beginFrame(m_frame);

            m_runtime->executor.execute(m_frame);

            //FrameMark;

            // Frame pacing
            auto frameEnd = std::chrono::steady_clock::now();

            std::chrono::duration<float> frameTime = frameEnd - frameStart;

            float sleepTime = kTargetFrameTime - frameTime.count();

            if (sleepTime > 0.0f)
            {
                std::this_thread::sleep_for(
                    std::chrono::duration<float>(sleepTime)
                );
            }

        }

        shutdown();
        return 0;
    }

    void AppBase::dispatchEvent(EventChannel channel, Event& e)
    {

        // channel based routing
        switch (channel)
        {
        case EventChannel::Input:
            handleInputEvent(e);
            break;

        case EventChannel::Window:
            handleWindowEvent(e);
            break;

        case EventChannel::Renderer:
            handleRenderEvent(e);
            break;
        }

    }

    void AppBase::handleInputEvent(Event& e)
    {

        if (e.type == EventType::KeyPressed &&
            e.key.key == IcKey::ESCAPE)
        {
            m_runtime->window->requestClose();
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
            m_runtime->window->requestClose();
            break;

        default:
            break;
        }
    }

    void AppBase::handleRenderEvent([[maybe_unused]] Event& e)
    {
        // reserved for render driven events later
    }

    void AppBase::createPlatform()
    {
        m_runtime->platform =
            std::make_unique<PlatformState>();
    }

    void AppBase::createJobSystem()
    {
        m_runtime->jobs = std::make_unique<JobSystem>();
        uint32_t coreCount = std::thread::hardware_concurrency();
        uint32_t workerCount = coreCount > 1 ? coreCount - 1 : 1;
        m_runtime->jobs->init(workerCount);

    }

    void AppBase::createWindow()
    {
        m_runtime->window =
            std::make_unique<GLFWWindow>(
                m_spec.window);
    }

    void AppBase::createInput()
    {
        m_runtime->input =
            std::make_unique<GLFWInput>(
                *m_runtime->window);
    }

    void AppBase::bindEvents()
    {
        m_runtime->window->bindEventSink(
            [this](const Event& e)
            {
                m_runtime->eventQueue->push(e);
            });
    }

    void AppBase::buildServices()
    {
        m_services =
        {
            m_runtime->input.get(),
            m_runtime->window.get(),
            m_runtime->jobs.get()
        };
    }

    void AppBase::shutdown()
    {
        onShutdown(); // client (DemoApp)

        if (m_runtime->jobs) m_runtime->jobs->shutdown();
    }
}