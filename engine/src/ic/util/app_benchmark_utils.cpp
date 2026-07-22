#include "ic/common/ic_pch.h"
#include "ic/util/app_benchmark_utils.h"

#include "ic/renderer/global_illumination/global_illumination.h"
#include "ic/renderer/renderer.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>

#include <spdlog/spdlog.h>

namespace ic::benchmark
{
    std::optional<std::string> environmentValue(const char* name)
    {
#ifdef _WIN32
        char* value = nullptr;
        size_t size = 0;
        if (_dupenv_s(&value, &size, name) != 0 || !value)
            return std::nullopt;
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
        const auto value = environmentValue(name);
        if (!value || value->empty())
            return std::nullopt;
        try
        {
            return std::stol(*value);
        }
        catch (const std::exception&)
        {
            spdlog::warn("[Benchmark] Ignoring non-numeric {}='{}'.",
                name, *value);
            return std::nullopt;
        }
    }

    BenchmarkConfig benchmarkConfigFromEnvironment()
    {
        BenchmarkConfig config{};
        if (const auto value = environmentInteger("IC_BENCH_FRAMES");
            value && *value > 0)
            config.frames = static_cast<uint32_t>(*value);
        if (const auto value = environmentInteger("IC_BENCH_WARMUP");
            value && *value >= 0)
            config.warmup = static_cast<uint32_t>(*value);
        if (const auto value = environmentInteger("IC_BENCH_WARMUP_S");
            value && *value >= 0)
            config.warmupSeconds = static_cast<double>(*value);
        if (const auto value = environmentInteger("IC_BENCH_ASYNC"))
            config.asyncOverride = *value != 0 ? 1 : 0;
        if (const auto value = environmentInteger("IC_BENCH_PROFILER"))
            config.profilerOverride = *value != 0 ? 1 : 0;
        if (const auto value = environmentInteger("IC_BENCH_DIAGNOSTICS"))
            config.diagnosticsMask = static_cast<int>(
                std::clamp(*value, 0l, 255l));
        if (const auto value = environmentInteger("IC_BENCH_TARGET_FPS");
            value && *value >= 0)
            config.targetFps = static_cast<float>(*value);
        if (const auto value = environmentValue("IC_BENCH_TAG"))
            config.tag = *value;
        if (const auto quality = environmentValue("IC_BENCH_GI_QUALITY"))
        {
            std::string value = *quality;
            std::ranges::transform(value, value.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (value == "off") config.giQuality = 0;
            else if (value == "low") config.giQuality = 1;
            else if (value == "medium") config.giQuality = 2;
            else if (value == "high") config.giQuality = 3;
            else if (value == "ultra") config.giQuality = 4;
            else spdlog::warn(
                "[Benchmark] Ignoring unknown IC_BENCH_GI_QUALITY='{}'.", value);
        }
        if (const auto value = environmentInteger("IC_BENCH_GI_CYCLE");
            value && *value > 0)
            config.giCycleInterval = static_cast<uint32_t>(*value);
        return config;
    }

    void BenchmarkAccumulator::add(
        double frameMs,
        double framePeriodMs,
        const GpuQueueTimelineStats& timeline,
        std::span<const GpuPassSample> samples,
        double frameUiBuildMs,
        const RendererPerformanceCounters& counters,
        const Renderer& renderer)
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
            PassAccumulator& pass = passes[renderer.passName(sample.node)];
            pass.totalMilliseconds += sample.endMs - sample.beginMs;
            pass.queue = sample.queue;
            ++pass.frames;
        }
    }

    double percentile(std::vector<double>& values, double fraction)
    {
        if (values.empty())
            return 0.0;
        const size_t index = std::min(values.size() - 1,
            static_cast<size_t>(fraction * (values.size() - 1)));
        std::nth_element(values.begin(), values.begin() + index, values.end());
        return values[index];
    }

    BenchmarkManager::BenchmarkManager()
        : m_config(benchmarkConfigFromEnvironment())
        , m_startTime(std::chrono::steady_clock::now())
    {
    }

    bool BenchmarkManager::shouldExit() const
    {
        return m_stats.frameMilliseconds.size() >= m_config.frames;
    }

    void BenchmarkManager::configureRenderer(Renderer& renderer) const
    {
        renderer.setVsyncEnabled(false);
        if (m_config.asyncOverride >= 0)
            renderer.setAsyncComputeEnabled(m_config.asyncOverride != 0);
        if (m_config.profilerOverride >= 0)
            renderer.setGpuProfilingEnabled(m_config.profilerOverride != 0);
        if (m_config.diagnosticsMask >= 0)
            renderer.setDiagnosticsSectionMask(
                static_cast<uint32_t>(m_config.diagnosticsMask));
        if (m_config.giQuality >= 0)
            renderer.setGlobalIlluminationQuality(
                static_cast<GlobalIlluminationQuality>(m_config.giQuality));
        spdlog::info(
            "[Benchmark] tag='{}' frames={} warmup={}f/{}s async={} "
            "profiler={} diagnosticsMask={} (vsync off, targetFps={})",
            m_config.tag, m_config.frames, m_config.warmup,
            m_config.warmupSeconds,
            renderer.asyncComputeEnabled() ? "on" : "off",
            renderer.gpuProfilingEnabled() ? "on" : "off",
            m_config.diagnosticsMask, m_config.targetFps);
    }

    void BenchmarkManager::processFrame(
        Renderer& renderer,
        uint32_t frameIndex,
        double framePeriodMs,
        double frameMs,
        const GpuQueueTimelineStats& timeline,
        std::span<const GpuPassSample> samples,
        double uiBuildMs,
        const RendererPerformanceCounters& counters)
    {
        if (m_config.giCycleInterval > 0u &&
            frameIndex % m_config.giCycleInterval == 0u)
        {
            const uint32_t preset =
                (frameIndex / m_config.giCycleInterval) % 5u;
            renderer.setGlobalIlluminationQuality(
                static_cast<GlobalIlluminationQuality>(preset));
            spdlog::info("[Benchmark] GI runtime switch frame={} quality={}",
                frameIndex, preset);
        }
        const double elapsedSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - m_startTime).count();
        if (frameIndex > m_config.warmup &&
            elapsedSeconds >= m_config.warmupSeconds)
        {
            m_stats.add(frameMs, framePeriodMs, timeline, samples,
                uiBuildMs, counters, renderer);
        }
    }

    void BenchmarkManager::finish(const Renderer& renderer)
    {
        reportBenchmark(m_config, m_stats, renderer);
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
            total += value;
        const double meanMs = total / static_cast<double>(count);
        const double meanPeriodMs =
            stats.fullFramePeriodMs / static_cast<double>(count);
        const double medianMs = percentile(stats.frameMilliseconds, 0.50);
        const double p95Ms = percentile(stats.frameMilliseconds, 0.95);
        const double timelineCount = stats.timelineSamples > 0
            ? static_cast<double>(stats.timelineSamples) : 1.0;
        const double graphicsMs = stats.graphicsBusyMs / timelineCount;
        const double computeMs = stats.computeBusyMs / timelineCount;
        const double overlapMs = stats.overlapMs / timelineCount;
        const FrameGraphTopology topology = renderer.frameGraphTopology();
        const GlobalIlluminationRuntimeStatistics gi =
            renderer.globalIlluminationStatistics();
        double giInclusiveMeanMs = 0.0;
        for (const auto& [name, pass] : stats.passes)
        {
            if (name.starts_with("GI."))
                giInclusiveMeanMs += pass.totalMilliseconds /
                    static_cast<double>(count);
        }
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
            config.tag, renderer.asyncComputeEnabled() ? "on" : "off", count,
            meanMs, meanPeriodMs,
            meanPeriodMs > 0.0 ? 1000.0 / meanPeriodMs : 0.0,
            medianMs, p95Ms, meanMs > 0.0 ? 1000.0 / meanMs : 0.0,
            graphicsMs, computeMs, overlapMs,
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
            topology.levels, topology.batches, topology.computeBatches,
            topology.crossQueueWaits);
        spdlog::info(
            "[Benchmark] GI tag='{}' quality={} active={} inclusive={:.4f}ms "
            "allocatedMiB={:.2f} probeMiB={:.2f} updates={} rays={} supported={}",
            config.tag, static_cast<uint32_t>(gi.quality), gi.active,
            giInclusiveMeanMs,
            static_cast<double>(gi.allocatedBytes) / (1024.0 * 1024.0),
            static_cast<double>(gi.probeBytes) / (1024.0 * 1024.0),
            gi.configuredProbeUpdates, gi.configuredRayBudget,
            gi.hardwareSupported);
        for (const auto& [name, pass] : stats.passes)
        {
            if (pass.frames == 0)
                continue;
            spdlog::info(
                "[Benchmark] PASS tag='{}' name='{}' queue={} "
                "meanMs={:.4f} executedFrames={}",
                config.tag, name,
                pass.queue == QueueType::Compute ? "compute" : "graphics",
                pass.totalMilliseconds / static_cast<double>(count),
                pass.frames);
        }
    }
}
