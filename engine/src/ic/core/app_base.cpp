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
#include "ic/core/frame_pipeline.h"
#include "ic/core/memory/frame_arena.h"
#include "ic/core/memory/frame_arena_manager.h"
#include "ic/renderer/renderer.h"
#include "ic/core/asset_manager.h"
#include "ic/scene/scene_manager.h"
#include "ic/core/debug_gui_layer.h"

#include <algorithm>
#include <filesystem>

#include <spdlog/spdlog.h>


namespace ic
{
    struct AppBase::PlatformState
    {
        GLFWContext glfw;
    };

    struct AppBase::AppRuntime
    {
        Scope<PlatformState>        platform;
        Scope<Window>               window;
        Scope<Input>                input;
        Scope<JobSystem>            jobs;
        Scope<EventQueue>           eventQueue;
        Scope<FrameArenaManager<AppSpecification::kMaxFramesInFlight>> 
                                    frameArenas;
        Scope<AssetManager>         assetManager;
        Scope<SceneManager>         sceneManager;
        Scope<Renderer>             renderer;
        
        EventFrameBuffer<AppSpecification::kMaxFramesInFlight> 
                                    eventBuffer;
        FramePipeline               framePipeline;

        AppRuntime(AppBase& app)
            : framePipeline(app)
            , eventQueue(std::make_unique<EventQueue>())
        {
        }
    };

    AppBase::AppBase(AppSpecification spec)
        : m_spec(std::move(spec))
        , m_runtime(std::make_unique<AppRuntime>(*this))
    {
    }

    AppBase::~AppBase() = default;

    void AppBase::initAppBase(
        [[maybe_unused]] int argc,
        [[maybe_unused]] char** argv)
    {

        spdlog::info("[AppBase] initAppBase...");

        createPlatform();
        createWindow();
        createInput();
        createJobSystem();
        createFrameArenas();
        createAssetManager();
        createSceneManager();


        createRenderer();

        bindEventSink();
        buildServices();

        initRenderer();

        if (m_spec.rendererSpec.useDebugGui)
        {
            pushLayer<DebugGuiLayer>();
        }
    }

    int AppBase::run(
        [[maybe_unused]] int argc,
        [[maybe_unused]] char** argv)
    {
        
        initAppBase(argc, argv);
        onInit(); // Client app configuration happens here


        m_clock.reset();
        m_frame.services = &m_services;
        
        auto& runtime = *m_runtime;

        spdlog::info("[AppBase] run... Entering main loop");

        // Main Loop
        while (!m_runtime->window->shouldClose())
        {
            ZoneScopedN("Frame");

            auto frameStart = std::chrono::steady_clock::now();

            ++m_frame.frameIndex;
            m_frame.deltaTime = m_clock.tick();
            m_frame.timeSinceStart = m_clock.getTimeSinceStart();
            m_frame.windowWidth = runtime.window->getWidth();
            m_frame.windowHeight = runtime.window->getHeight();
            
            auto& arena = m_runtime->frameArenas->beginFrame(m_frame.frameIndex);
            arena.reset();
            m_frame.arena = &arena;

            runtime.input->beginFrame();
            runtime.window->pollEvents();
            runtime.eventBuffer.ingest(*m_runtime->eventQueue);
            runtime.eventBuffer.beginFrame(m_frame);

            runtime.framePipeline.execute(m_frame);
            runtime.assetManager->update();
            runtime.sceneManager->update(m_frame);

            if (m_spec.rendererSpec.useDebugGui &&
                runtime.renderer->beginDebugGuiFrame())
            {
                m_layerStack.renderAll(m_frame.interpolationAlpha);
                runtime.renderer->endDebugGuiFrame();
            }

            runtime.renderer->render(
                m_frame,
                runtime.sceneManager->renderView());

            // Frame pacing
            sleep(frameStart);

            FrameMark;
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
        if (m_runtime->input)
        {
            m_runtime->input->onEvent(e);
        }

        if (e.type == EventType::KeyPressed && e.key.key == IcKey::ESCAPE)
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

    void AppBase::sleep(const auto& frameStart)
    {
        auto frameEnd = std::chrono::steady_clock::now();
        std::chrono::duration<float> frameTime = frameEnd - frameStart;
        const float targetFps =
            m_spec.rendererSpec.settings.targetFps > 0.0f
                ? m_spec.rendererSpec.settings.targetFps
                : kFallbackTargetFPS;
        float sleepTime = (1.0f / targetFps) - frameTime.count();
        if (sleepTime > 0.0f)
        {
            std::this_thread::sleep_for(
                std::chrono::duration<float>(sleepTime)
            );
        }
    }

    void AppBase::createPlatform()
    {
        m_runtime->platform =
            std::make_unique<PlatformState>();
    }

    void AppBase::createJobSystem()
    {
        m_runtime->jobs = std::make_unique<JobSystem>();
        m_runtime->jobs->init();
    }

    void AppBase::createFrameArenas()
    {
        m_runtime->frameArenas =
            std::make_unique<FrameArenaManager<AppSpecification::kMaxFramesInFlight>>
            (16 * 1024 * 1024);
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

    void AppBase::createRenderer()
    {
        m_runtime->renderer = std::make_unique<Renderer>(
            m_spec.rendererSpec);
    }

    void AppBase::createAssetManager()
    {
        m_runtime->assetManager = std::make_unique<AssetManager>();

        AssetManagerDesc desc{};
        desc.assetRoot = m_spec.resourceRoots.assetRoot;
        desc.maxConcurrentLoads = std::max(1u, m_runtime->jobs->workerCount());

        m_runtime->assetManager->init(desc, *m_runtime->jobs);
    }

    void AppBase::createSceneManager()
    {
        m_runtime->sceneManager = std::make_unique<SceneManager>();

        SceneManagerDesc desc{};
        desc.enableAsyncSceneLoading = true;
        desc.modelRoot = m_spec.resourceRoots.modelRoot;
        desc.defaultEnvironment = m_spec.rendererSpec.settings.environment;

        m_runtime->sceneManager->init(
            desc,
            *m_runtime->assetManager,
            *m_runtime->jobs);
    }

    void AppBase::initRenderer()
    {
        auto workerCount = m_runtime->jobs->workerCount();

        m_runtime->renderer->init(
            m_spec.rendererSpec,
            *m_runtime->window,
            workerCount);
    }

    void AppBase::bindEventSink()
    {
        m_runtime->window->bindEventSink(
            [this](const Event& e)
            {
                m_runtime->eventQueue->push(e);
            });
    }

    void AppBase::buildServices()
    {
        m_services.input = m_runtime->input.get();
        m_services.window = m_runtime->window.get();
        m_services.renderer = m_runtime->renderer.get();

        m_services.jobSystem = m_runtime->jobs.get();
        m_services.assetManager = m_runtime->assetManager.get();
        m_services.sceneManager = m_runtime->sceneManager.get();
    }

    void AppBase::shutdown()
    {
        if (m_runtime->renderer)
            m_runtime->renderer->shutdown();

        if (m_runtime->sceneManager)
            m_runtime->sceneManager->shutdown();

        if (m_runtime->assetManager)
            m_runtime->assetManager->shutdown();

        if (m_runtime->jobs)
            m_runtime->jobs->shutdown();

        onShutdown(); // client (DemoApp)
    }
}
