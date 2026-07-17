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
#include "ic/core/frame_memory/frame_arena.h"
#include "ic/core/frame_memory/frame_arena_manager.h"
#include "ic/renderer/renderer.h"
#include "ic/core/asset_manager.h"
#include "ic/scene/scene_manager.h"
#include "ic/core/debug_gui_layer.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <spdlog/spdlog.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace ic
{
    namespace
    {
        // Blocks the calling thread for `seconds` accurately and without
        // busy-waiting. On Windows this uses a high-resolution waitable timer
        // (Win10 1803+) so the wait is precise to well under a millisecond
        // WITHOUT globally raising the system timer resolution (which would hurt
        // power use); it falls back to std::this_thread::sleep_for otherwise.
        void preciseSleepSeconds(double seconds)
        {
            if (seconds <= 0.0)
            {
                return;
            }
#ifdef _WIN32
#ifdef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
            static HANDLE timer = CreateWaitableTimerExW(
                nullptr, nullptr,
                CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                TIMER_ALL_ACCESS);
            if (timer)
            {
                LARGE_INTEGER due{};
                // Negative == relative, in 100 ns units.
                due.QuadPart = -static_cast<LONGLONG>(seconds * 1.0e7);
                if (SetWaitableTimerEx(
                        timer, &due, 0, nullptr, nullptr, nullptr, 0))
                {
                    WaitForSingleObject(timer, INFINITE);
                    return;
                }
            }
#endif
#endif
            std::this_thread::sleep_for(
                std::chrono::duration<double>(seconds));
        }

        std::optional<std::string> environmentValue(const char* name)
        {
#ifdef _WIN32
            char* value = nullptr;
            size_t size = 0;
            if (_dupenv_s(&value, &size, name) != 0 || !value)
            {
                return std::nullopt;
            }
            std::string result(value);
            free(value);
            return result;
#else
            const char* value = std::getenv(name);
            return value ? std::optional<std::string>(value) : std::nullopt;
#endif
        }

        std::optional<long> environmentInteger(const char* name)
        {
            const std::optional<std::string> value = environmentValue(name);
            if (!value || value->empty())
            {
                return std::nullopt;
            }
            try
            {
                return std::stol(*value);
            }
            catch (const std::exception&)
            {
                spdlog::warn(
                    "[AppBase] Ignoring non-numeric {}='{}'.",
                    name, *value);
                return std::nullopt;
            }
        }

        // Headless-ish measurement harness. The app still opens a window (the backends
        // render through a swapchain), but it runs a fixed number of frames
        // with no user input and prints one machine-readable summary line, so
        // async-compute on/off can be compared without a human at the keyboard.
        //
        //   IC_BENCH_FRAMES   measured frames; 0/unset disables the harness
        //   IC_BENCH_WARMUP   frames discarded first (PSO warm-up, swapchain
        //                     settling); default 200
        //   IC_BENCH_WARMUP_S seconds discarded first; default 6. This one is
        //                     load-bearing: models stream in asynchronously, so
        //                     an empty scene renders thousands of trivial frames
        //                     per second and a frame-only warmup is consumed
        //                     long before there is anything to draw. Both the
        //                     frame and the time threshold must pass.
        //   IC_BENCH_ASYNC    1/0 forces the async-compute toggle, unset leaves
        //                     whatever the renderer defaults to
        //   IC_BENCH_PROFILER 1/0 enables/disables GPU timestamp collection
        //   IC_BENCH_DIAGNOSTICS bit mask for open diagnostics sections;
        //                     0 closes the window, 255 opens all sections
        //   IC_BENCH_GUI      1/0 creates or completely disables ImGui
        //   IC_BENCH_TARGET_FPS CPU limiter target; 0 disables it
        //   IC_BENCH_TAG      free-form label echoed into the summary line
        struct BenchmarkConfig
        {
            uint32_t frames = 0;
            uint32_t warmup = 200;
            double warmupSeconds = 6.0;
            int asyncOverride = -1;
            int profilerOverride = -1;
            int diagnosticsMask = -1;
            float targetFps = 0.0f;
            std::string tag;

            [[nodiscard]] bool active() const { return frames > 0; }
        };

        BenchmarkConfig benchmarkConfigFromEnvironment()
        {
            BenchmarkConfig config{};
            if (const std::optional<long> frames =
                    environmentInteger("IC_BENCH_FRAMES");
                frames && *frames > 0)
            {
                config.frames = static_cast<uint32_t>(*frames);
            }
            if (const std::optional<long> warmup =
                    environmentInteger("IC_BENCH_WARMUP");
                warmup && *warmup >= 0)
            {
                config.warmup = static_cast<uint32_t>(*warmup);
            }
            if (const std::optional<long> warmupSeconds =
                    environmentInteger("IC_BENCH_WARMUP_S");
                warmupSeconds && *warmupSeconds >= 0)
            {
                config.warmupSeconds = static_cast<double>(*warmupSeconds);
            }
            if (const std::optional<long> async =
                    environmentInteger("IC_BENCH_ASYNC"))
            {
                config.asyncOverride = *async != 0 ? 1 : 0;
            }
            if (const std::optional<long> profiler =
                    environmentInteger("IC_BENCH_PROFILER"))
            {
                config.profilerOverride = *profiler != 0 ? 1 : 0;
            }
            if (const std::optional<long> diagnostics =
                    environmentInteger("IC_BENCH_DIAGNOSTICS"))
            {
                config.diagnosticsMask = static_cast<int>(
                    std::clamp(*diagnostics, 0l, 255l));
            }
            if (const std::optional<long> targetFps =
                    environmentInteger("IC_BENCH_TARGET_FPS");
                targetFps && *targetFps >= 0)
            {
                config.targetFps = static_cast<float>(*targetFps);
            }
            if (const std::optional<std::string> tag =
                    environmentValue("IC_BENCH_TAG"))
            {
                config.tag = *tag;
            }
            return config;
        }

        struct PassAccumulator
        {
            double totalMilliseconds = 0.0;
            uint64_t frames = 0;
            QueueType queue = QueueType::Graphics;
        };

        struct BenchmarkAccumulator
        {
            std::vector<double> frameMilliseconds;
            double graphicsBusyMs = 0.0;
            double computeBusyMs = 0.0;
            double overlapMs = 0.0;
            uint64_t timelineSamples = 0;
            double uiBuildMs = 0.0;
            double fullFramePeriodMs = 0.0;
            RendererPerformanceCounters performance{};
            std::map<GraphNodeId, PassAccumulator> passes;

            void add(
                double frameMs,
                double framePeriodMs,
                const GpuQueueTimelineStats& timeline,
                std::span<const GpuPassSample> samples,
                double frameUiBuildMs,
                const RendererPerformanceCounters& counters)
            {
                frameMilliseconds.push_back(frameMs);
                fullFramePeriodMs += framePeriodMs;
                uiBuildMs += frameUiBuildMs;
                performance.backendCpuMs += counters.backendCpuMs;
                performance.frameSlotWaitMs += counters.frameSlotWaitMs;
                performance.profilerReadbackMs += counters.profilerReadbackMs;
                performance.graphRecordMs += counters.graphRecordMs;
                performance.uiRecordMs += counters.uiRecordMs;
                performance.validationMs += counters.validationMs;
                performance.submitPresentMs += counters.submitPresentMs;
                if (timeline.valid)
                {
                    graphicsBusyMs += timeline.graphicsBusyMs;
                    computeBusyMs += timeline.computeBusyMs;
                    overlapMs += timeline.overlapMs;
                    ++timelineSamples;
                }

                for (const GpuPassSample& sample : samples)
                {
                    PassAccumulator& pass = passes[sample.node];
                    pass.totalMilliseconds += sample.endMs - sample.beginMs;
                    pass.queue = sample.queue;
                    ++pass.frames;
                }
            }
        };

        double percentile(std::vector<double>& values, double fraction)
        {
            if (values.empty())
            {
                return 0.0;
            }
            const size_t index = std::min(
                values.size() - 1,
                static_cast<size_t>(fraction * (values.size() - 1)));
            std::nth_element(
                values.begin(), values.begin() + index, values.end());
            return values[index];
        }

        void reportBenchmark(
            const BenchmarkConfig& config,
            BenchmarkAccumulator& stats,
            const Renderer& renderer)
        {
            if (stats.frameMilliseconds.empty())
            {
                spdlog::warn("[Benchmark] No frames measured.");
                return;
            }

            const size_t count = stats.frameMilliseconds.size();
            double total = 0.0;
            for (const double value : stats.frameMilliseconds)
            {
                total += value;
            }
            const double meanMs = total / static_cast<double>(count);
            const double meanPeriodMs =
                stats.fullFramePeriodMs / static_cast<double>(count);
            const double medianMs = percentile(stats.frameMilliseconds, 0.50);
            const double p95Ms = percentile(stats.frameMilliseconds, 0.95);

            const double timelineCount =
                stats.timelineSamples > 0
                    ? static_cast<double>(stats.timelineSamples)
                    : 1.0;
            const double graphicsMs = stats.graphicsBusyMs / timelineCount;
            const double computeMs = stats.computeBusyMs / timelineCount;
            const double overlapMs = stats.overlapMs / timelineCount;

            const FrameGraphTopology topology = renderer.frameGraphTopology();
            spdlog::info(
                "[Benchmark] RESULT tag='{}' async={} frames={} "
                "workMean={:.3f}ms fullPeriod={:.3f}ms fullFps={:.1f} "
                "median={:.3f}ms p95={:.3f}ms workFps={:.1f} "
                "gpuGraphics={:.3f}ms gpuCompute={:.3f}ms overlap={:.3f}ms "
                "overlapPctOfCompute={:.1f} timelineFrames={} "
                "cpuBackend={:.3f}ms frameWait={:.3f}ms "
                "profilerReadback={:.4f}ms graphRecord={:.3f}ms "
                "uiBuild={:.3f}ms uiRecord={:.4f}ms validation={:.4f}ms "
                "submitPresent={:.3f}ms levels={} batches={} "
                "computeBatches={} crossQueueWaits={}",
                config.tag,
                renderer.asyncComputeEnabled() ? "on" : "off",
                count,
                meanMs,
                meanPeriodMs,
                meanPeriodMs > 0.0 ? 1000.0 / meanPeriodMs : 0.0,
                medianMs,
                p95Ms,
                meanMs > 0.0 ? 1000.0 / meanMs : 0.0,
                graphicsMs,
                computeMs,
                overlapMs,
                computeMs > 0.0 ? 100.0 * overlapMs / computeMs : 0.0,
                stats.timelineSamples,
                stats.performance.backendCpuMs / static_cast<double>(count),
                stats.performance.frameSlotWaitMs / static_cast<double>(count),
                stats.performance.profilerReadbackMs / static_cast<double>(count),
                stats.performance.graphRecordMs / static_cast<double>(count),
                stats.uiBuildMs / static_cast<double>(count),
                stats.performance.uiRecordMs / static_cast<double>(count),
                stats.performance.validationMs / static_cast<double>(count),
                stats.performance.submitPresentMs / static_cast<double>(count),
                topology.levels,
                topology.batches,
                topology.computeBatches,
                topology.crossQueueWaits);

            for (const auto& [node, pass] : stats.passes)
            {
                if (pass.frames == 0)
                {
                    continue;
                }
                // Averaged over measured frames, not over frames the pass ran:
                // a cadence-gated pass that executes rarely should show a small
                // per-frame cost, because that is what it actually costs.
                spdlog::info(
                    "[Benchmark] PASS tag='{}' name='{}' queue={} "
                    "meanMs={:.4f} executedFrames={}",
                    config.tag,
                    renderer.passName(node),
                    pass.queue == QueueType::Compute ? "compute" : "graphics",
                    pass.totalMilliseconds / static_cast<double>(count),
                    pass.frames);
            }
        }
    }

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
        {}
    };

    AppBase::AppBase(AppSpecification spec)
        : m_spec(std::move(spec))
        , m_runtime(std::make_unique<AppRuntime>(*this))
    {}

    AppBase::~AppBase() = default;

    void AppBase::initAppBase(
        [[maybe_unused]] int argc,
        [[maybe_unused]] char** argv)
    {

        if (const std::optional<long> benchmarkGui =
                environmentInteger("IC_BENCH_GUI");
            benchmarkGui && *benchmarkGui == 0)
        {
            m_spec.rendererSpec.useDebugGui = false;
        }
        if (const std::optional<long> benchmarkValidation =
                environmentInteger("IC_BENCH_VALIDATION"))
        {
            m_spec.rendererSpec.enableValidation = *benchmarkValidation != 0;
        }

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

        const BenchmarkConfig benchmark = benchmarkConfigFromEnvironment();
        BenchmarkAccumulator benchmarkStats{};
        const auto benchmarkStart = std::chrono::steady_clock::now();
        if (benchmark.active())
        {
            // Throughput has to be measured against the GPU, not against a
            // limiter: both the frame cap and vsync would clamp every variant
            // to the same number and hide the effect under test.
            m_spec.rendererSpec.settings.targetFps = benchmark.targetFps;
            runtime.renderer->setVsyncEnabled(false);
            if (benchmark.asyncOverride >= 0)
            {
                runtime.renderer->setAsyncComputeEnabled(
                    benchmark.asyncOverride != 0);
            }
            if (benchmark.profilerOverride >= 0)
            {
                runtime.renderer->setGpuProfilingEnabled(
                    benchmark.profilerOverride != 0);
            }
            if (benchmark.diagnosticsMask >= 0)
            {
                runtime.renderer->setDiagnosticsSectionMask(
                    static_cast<uint32_t>(benchmark.diagnosticsMask));
            }
            spdlog::info(
                "[Benchmark] tag='{}' frames={} warmup={}f/{}s async={} "
                "profiler={} diagnosticsMask={} gui={} "
                "(vsync off, targetFps={})",
                benchmark.tag, benchmark.frames, benchmark.warmup,
                benchmark.warmupSeconds,
                runtime.renderer->asyncComputeEnabled() ? "on" : "off",
                runtime.renderer->gpuProfilingEnabled() ? "on" : "off",
                benchmark.diagnosticsMask,
                m_spec.rendererSpec.useDebugGui ? "on" : "off",
                benchmark.targetFps);
        }

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

            const auto uiStart = std::chrono::steady_clock::now();
            if (m_spec.rendererSpec.useDebugGui &&
                runtime.renderer->beginDebugGuiFrame())
            {
                m_layerStack.renderAll(m_frame.interpolationAlpha);
                runtime.renderer->endDebugGuiFrame();
            }
            const double uiBuildMs =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - uiStart).count();

            runtime.renderer->render(
                m_frame,
                runtime.sceneManager->renderView());

            if (benchmark.active())
            {
                // Sampled before the pacing sleep so this is the real cost of
                // producing a frame, not the limiter's period.
                const auto now = std::chrono::steady_clock::now();
                const double frameMs =
                    std::chrono::duration<double, std::milli>(
                        now - frameStart).count();
                const double elapsedSeconds =
                    std::chrono::duration<double>(now - benchmarkStart).count();
                if (m_frame.frameIndex > benchmark.warmup &&
                    elapsedSeconds >= benchmark.warmupSeconds)
                {
                    benchmarkStats.add(
                        frameMs,
                        m_frame.deltaTime * 1000.0,
                        runtime.renderer->gpuQueueTimeline(),
                        runtime.renderer->gpuPassSamples(),
                        uiBuildMs,
                        runtime.renderer->performanceCounters());
                }
                if (benchmarkStats.frameMilliseconds.size() >= benchmark.frames)
                {
                    reportBenchmark(
                        benchmark, benchmarkStats, *runtime.renderer);
                    break;
                }
            }

            // Frame pacing
            sleep(frameStart);

            FrameMark;
        }

        shutdown();
        return 0;
    }

    void AppBase::dispatchEvent(EventChannel channel, Event& e)
    {
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

        if (e.type == EventType::KeyPressed)
        {
            const KeyEvent* key = getPayload<KeyEvent>(e);

            if (key && key->key == IcKey::ESCAPE)
            {
                m_runtime->window->requestClose();
            }
        }

        if (e.type == EventType::MouseButtonPressed)
        {
            const MouseButtonEvent* mouseButton =
                getPayload<MouseButtonEvent>(e);

            if (mouseButton)
            {
                spdlog::info(
                    "Mouse button: {}",
                    static_cast<int>(mouseButton->button));
            }
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
        // targetFps <= 0 means "no cap": present pacing (vsync) and/or the GPU
        // are the only limits, so we never sleep. Otherwise cap the frame rate
        // to targetFps by sleeping off only the remaining time. When the frame
        // already took at least the target period (GPU/present-bound) the sleep
        // is skipped, so this never adds a stall to a frame that is already at
        // or below the target rate.
        const float targetFps = m_spec.rendererSpec.settings.targetFps;
        if (targetFps <= 0.0f)
        {
            return;
        }

        const auto frameEnd = std::chrono::steady_clock::now();
        const std::chrono::duration<double> frameTime = frameEnd - frameStart;
        const double sleepTime =
            (1.0 / static_cast<double>(targetFps)) - frameTime.count();
        preciseSleepSeconds(sleepTime);
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
