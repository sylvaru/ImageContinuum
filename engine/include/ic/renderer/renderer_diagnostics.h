// ic/renderer/renderer_diagnostics.h
#pragma once
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "gpu_queue_profiler.h"

namespace ic
{
    class Renderer;

    // A bounded, preallocated ring of per-frame samples plus the wall time each
    // one covered, so statistics can be taken over a time window rather than a
    // sample count (a "last 128 samples" window means 2 seconds at 60 fps and
    // 0.12 s at 1000 fps, which makes the numbers unreadable across scenes).
    //
    // Sampling is deliberately raw: every frame's value goes in unsmoothed.
    // Smoothing happens only when statistics are read, so nothing that consumes
    // this can be fed a filtered value by accident.
    class DiagnosticSeries
    {
    public:
        struct Stats
        {
            float current = 0.0f;
            float average = 0.0f;
            float minimum = 0.0f;
            float maximum = 0.0f;
            float p95 = 0.0f;
            uint32_t samples = 0;
            bool valid = false;
        };

        // Allocates once; push() never allocates afterwards.
        void init(uint32_t capacity);
        void push(float value, float deltaSeconds);
        void clear();

        [[nodiscard]] bool empty() const { return m_count == 0; }
        [[nodiscard]] float current() const;

        // Walks back over at most windowSeconds of history. `scratch` is
        // supplied by the caller and reused across series so the percentile
        // sort does not allocate per call.
        [[nodiscard]] Stats stats(
            float windowSeconds, std::vector<float>& scratch) const;

    private:
        [[nodiscard]] uint32_t indexFromNewest(uint32_t age) const;

        std::vector<float> m_values;
        std::vector<float> m_deltas;
        uint32_t m_head = 0;   // next write slot
        uint32_t m_count = 0;
    };

    // The consolidated "Renderer Diagnostics" window: one expandable window
    // covering frame/GPU timing, the async-compute decision, queue occupancy,
    // the frame graph, per-pass timings, resources, culling and backend
    // capabilities.
    //
    // Two rules this type exists to enforce:
    //  1. It is a pure OBSERVER. It never feeds anything back into the
    //     renderer's scheduling, and in particular never into
    //     renderer scheduling, which consumes raw per-frame samples
    //     directly. Smoothed/paused values live here and go no further.
    //  2. It costs ~nothing when it is not being looked at: sampling is skipped
    //     when the debug GUI is off, and per-section work (percentile sorts,
    //     string building, table rows) only runs while that section is open.
    class RendererDiagnostics
    {
    public:
        // Per-frame measurements, taken raw. Mirrors what the async policy is
        // given, so the two can never silently diverge in meaning.
        struct FrameSample
        {
            float deltaSeconds = 0.0f;
            float frameMs = 0.0f;
            GpuQueueTimelineStats timeline;
            std::span<const GpuPassSample> passSamples;
        };

        RendererDiagnostics();

        // Cheap and allocation-free after the first call. Skipped entirely when
        // the debug GUI is disabled.
        void sample(const FrameSample& frame);

        // Draws the window. `renderer` is used both to read state and to apply
        // the controls the window hosts (vsync, async mode, cull debug views).
        void draw(Renderer& renderer);

        [[nodiscard]] bool windowOpen() const { return m_windowOpen; }
        void setWindowOpen(bool open) { m_windowOpen = open; }
        void setSectionOpenMask(uint32_t mask)
        {
            m_sectionOpenMask = mask & 0xffu;
            m_displayDirty = true;
        }

    private:
        // Section identifiers, used to skip building anything a collapsed
        // section would have shown.
        enum class Section : uint32_t
        {
            Overview,
            AsyncCompute,
            GpuQueues,
            FrameGraph,
            PassTimings,
            Resources,
            Visibility,
            Backend,
            Count
        };

        struct PassSeries
        {
            DiagnosticSeries series;
            GraphNodeId node{};
            QueueType queue = QueueType::Graphics;
            bool active = false;
        };

        void drawControls();
        void drawOverview(Renderer& renderer);
        void drawAsyncCompute(Renderer& renderer);
        void drawGpuQueues(Renderer& renderer);
        void drawFrameGraph(Renderer& renderer);
        void drawPassTimings(Renderer& renderer);
        void drawResources(Renderer& renderer);
        void drawVisibility(Renderer& renderer);
        void drawBackend(Renderer& renderer);

        // Wraps ImGui::CollapsingHeader and records whether the section is open
        // so sample-side work can be skipped for closed sections.
        bool beginSection(Section section, const char* label);
        [[nodiscard]] bool sectionOpen(Section section) const;

        void refreshDisplayStats();
        void resetHistory();

        // Draws "current | avg | min | max | p95" for one series.
        void drawSeriesRow(
            const char* label,
            const DiagnosticSeries::Stats& stats,
            const char* unit,
            const char* tooltip);

        static void helpMarker(const char* text);

        bool m_windowOpen = true;
        bool m_windowVisible = true;
        bool m_paused = false;

        // Display refresh rate, clamped to 4-10 Hz. Fast enough to feel live,
        // slow enough that the digits can actually be read.
        float m_displayHz = 6.0f;
        float m_displayAccumulator = 0.0f;
        bool m_displayDirty = true;

        // Window of history the statistics cover, in seconds.
        float m_historySeconds = 2.0f;

        DiagnosticSeries m_frameMs;
        DiagnosticSeries m_gpuGraphicsMs;
        DiagnosticSeries m_gpuComputeMs;
        DiagnosticSeries m_gpuOverlapMs;
        DiagnosticSeries m_gpuBusyMs;

        // Cached statistics, recomputed only at the display rate.
        DiagnosticSeries::Stats m_frameStats;
        DiagnosticSeries::Stats m_gpuGraphicsStats;
        DiagnosticSeries::Stats m_gpuComputeStats;
        DiagnosticSeries::Stats m_gpuOverlapStats;
        DiagnosticSeries::Stats m_gpuBusyStats;

        std::vector<PassSeries> m_passes;
        std::vector<DiagnosticSeries::Stats> m_passStats;

        // Reused by the percentile computation; never resized after warmup.
        std::vector<float> m_scratch;

        // Reused by the frame-graph wait column so the table does not build
        // std::strings every row every frame.
        std::string m_textScratch;

        // Per-level queue-batch counts, rebuilt into existing storage.
        std::vector<uint32_t> m_levelQueueCounts;

        uint32_t m_sectionOpenMask = 0;

        // Pass-timing table controls.
        int m_passSortColumn = 2;
        bool m_passSortDescending = true;
        std::vector<char> m_passFilter;
        std::vector<uint32_t> m_passOrder;
    };
}
