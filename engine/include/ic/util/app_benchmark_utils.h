#pragma once

#include <chrono>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "ic/renderer/frame_graph/frame_graph_types.h"
#include "ic/renderer/render_types.h"
#include "ic/renderer/renderer_backend.h"

namespace ic
{
    class Renderer;
}

namespace ic::benchmark
{
    std::optional<std::string> environmentValue(const char* name);
    std::optional<long> environmentInteger(const char* name);

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
    //   IC_BENCH_GI_QUALITY off/low/medium/high/ultra; unset keeps config
    //   IC_BENCH_GI_CYCLE  frames between Off/Low/Medium/High/Ultra switches
    struct BenchmarkConfig
    {
        uint32_t frames = 0;
        uint32_t warmup = 200;
        double warmupSeconds = 6.0;
        int asyncOverride = -1;
        int profilerOverride = -1;
        int diagnosticsMask = -1;
        int giQuality = -1;
        uint32_t giCycleInterval = 0;
        float targetFps = 0.0f;
        std::string tag;

        [[nodiscard]] bool active() const { return frames > 0; }
    };

    BenchmarkConfig benchmarkConfigFromEnvironment();

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
        std::map<std::string, PassAccumulator, std::less<>> passes;

        void add(
            double frameMs,
            double framePeriodMs,
            const GpuQueueTimelineStats& timeline,
            std::span<const GpuPassSample> samples,
            double frameUiBuildMs,
            const RendererPerformanceCounters& counters,
            const Renderer& renderer);
    };

    double percentile(std::vector<double>& values, double fraction);

    void reportBenchmark(
        const BenchmarkConfig& config,
        BenchmarkAccumulator& stats,
        const Renderer& renderer);

    class BenchmarkManager
    {
    public:
        BenchmarkManager();

        bool isActive() const { return m_config.active(); }
        bool shouldExit() const;
        float getTargetFps() const { return m_config.targetFps; }
        void configureRenderer(Renderer& renderer) const;
        void processFrame(
            Renderer& renderer,
            uint32_t frameIndex,
            double framePeriodMs,
            double frameMs,
            const GpuQueueTimelineStats& timeline,
            std::span<const GpuPassSample> samples,
            double uiBuildMs,
            const RendererPerformanceCounters& counters);
        void finish(const Renderer& renderer);

    private:
        BenchmarkConfig m_config;
        BenchmarkAccumulator m_stats;
        std::chrono::steady_clock::time_point m_startTime;
    };

}
