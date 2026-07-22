#include "ic/renderer/renderer_diagnostics.h"

#include "ic/renderer/renderer.h"
#include "ic/renderer/renderer_backend.h"
#include "ic/renderer/frame_graph/compiled_graph_plan.h"

#include <imgui.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace ic
{
    namespace
    {
        // Per-frame samples retained per series. At ~1000 fps this is ~1 s of
        // history, at 60 fps ~17 s; the window control selects a slice of it.
        // Allocated once per series and never grown.
        constexpr uint32_t kFrameHistoryCapacity = 1024;
        // Per-pass history is shorter: there can be one series per pass, and a
        // 2 s window at any realistic frame rate fits comfortably.
        constexpr uint32_t kPassHistoryCapacity = 256;
        constexpr uint32_t kMaxTrackedPasses = 128;

        const ImVec4 kGreen(0.25f, 0.9f, 0.35f, 1.0f);
        const ImVec4 kAmber(1.0f, 0.8f, 0.35f, 1.0f);
        const ImVec4 kOrange(1.0f, 0.65f, 0.25f, 1.0f);
        const ImVec4 kBlue(0.6f, 0.7f, 1.0f, 1.0f);
        const ImVec4 kRed(1.0f, 0.3f, 0.25f, 1.0f);

        const char* queueName(QueueType queue)
        {
            switch (queue)
            {
            case QueueType::Graphics: return "Graphics";
            case QueueType::Compute:  return "Compute";
            case QueueType::Transfer: return "Transfer";
            }
            return "Unknown";
        }

        const char* renderPathName(RenderPathType path)
        {
            switch (path)
            {
            case RenderPathType::Forward:          return "Forward";
            case RenderPathType::ClusteredForward: return "Clustered Forward";
            case RenderPathType::PathTraced:       return "Path Traced";
            }
            return "Unknown";
        }

        const char* ownershipName(ResourceOwnership ownership)
        {
            switch (ownership)
            {
            case ResourceOwnership::Transient:  return "Transient";
            case ResourceOwnership::Persistent: return "Persistent";
            case ResourceOwnership::Imported:   return "Imported";
            }
            return "Unknown";
        }

        const char* multiplicityName(ResourceMultiplicity multiplicity)
        {
            switch (multiplicity)
            {
            case ResourceMultiplicity::Single:       return "Single";
            case ResourceMultiplicity::PerFrameSlot: return "Per-frame-slot";
            case ResourceMultiplicity::History:      return "History";
            }
            return "Unknown";
        }

        void formatBytes(uint64_t bytes, char* out, size_t size)
        {
            if (bytes >= (1ull << 20))
            {
                std::snprintf(
                    out, size, "%.1f MB",
                    static_cast<double>(bytes) / static_cast<double>(1ull << 20));
            }
            else if (bytes >= (1ull << 10))
            {
                std::snprintf(
                    out, size, "%.1f KB",
                    static_cast<double>(bytes) / static_cast<double>(1ull << 10));
            }
            else
            {
                std::snprintf(out, size, "%llu B",
                    static_cast<unsigned long long>(bytes));
            }
        }
    }

    void DiagnosticSeries::init(uint32_t capacity)
    {
        m_values.assign(std::max(1u, capacity), 0.0f);
        m_deltas.assign(std::max(1u, capacity), 0.0f);
        m_head = 0;
        m_count = 0;
    }

    void DiagnosticSeries::push(float value, float deltaSeconds)
    {
        if (m_values.empty())
        {
            return;
        }
        m_values[m_head] = value;
        m_deltas[m_head] = deltaSeconds;
        m_head = (m_head + 1u) % static_cast<uint32_t>(m_values.size());
        m_count = std::min(
            m_count + 1u, static_cast<uint32_t>(m_values.size()));
    }

    void DiagnosticSeries::clear()
    {
        m_head = 0;
        m_count = 0;
    }

    uint32_t DiagnosticSeries::indexFromNewest(uint32_t age) const
    {
        const uint32_t capacity = static_cast<uint32_t>(m_values.size());
        return (m_head + capacity - 1u - age) % capacity;
    }

    float DiagnosticSeries::current() const
    {
        return m_count == 0 ? 0.0f : m_values[indexFromNewest(0)];
    }

    DiagnosticSeries::Stats DiagnosticSeries::stats(
        float windowSeconds, std::vector<float>& scratch) const
    {
        Stats result{};
        if (m_count == 0)
        {
            return result;
        }

        scratch.clear();
        float elapsed = 0.0f;
        float sum = 0.0f;
        result.current = m_values[indexFromNewest(0)];
        result.minimum = result.current;
        result.maximum = result.current;

        // Walk newest -> oldest until the requested window is covered. Always
        // takes at least one sample so a fresh series still reads sensibly.
        for (uint32_t age = 0; age < m_count; ++age)
        {
            const uint32_t index = indexFromNewest(age);
            const float value = m_values[index];

            scratch.push_back(value);
            sum += value;
            result.minimum = std::min(result.minimum, value);
            result.maximum = std::max(result.maximum, value);

            elapsed += m_deltas[index];
            if (elapsed >= windowSeconds)
            {
                break;
            }
        }

        result.samples = static_cast<uint32_t>(scratch.size());
        result.average = sum / static_cast<float>(result.samples);

        // Nth-element rather than a full sort: only the p95 position matters.
        const size_t rank = std::min(
            scratch.size() - 1u,
            static_cast<size_t>(
                std::ceil(0.95 * static_cast<double>(result.samples))) - 1u);
        std::nth_element(
            scratch.begin(), scratch.begin() + rank, scratch.end());
        result.p95 = scratch[rank];
        result.valid = true;
        return result;
    }

    RendererDiagnostics::RendererDiagnostics()
    {
        m_frameMs.init(kFrameHistoryCapacity);
        m_gpuGraphicsMs.init(kFrameHistoryCapacity);
        m_gpuComputeMs.init(kFrameHistoryCapacity);
        m_gpuOverlapMs.init(kFrameHistoryCapacity);
        m_gpuBusyMs.init(kFrameHistoryCapacity);

        m_passes.resize(kMaxTrackedPasses);
        for (PassSeries& pass : m_passes)
        {
            pass.series.init(kPassHistoryCapacity);
        }
        m_passStats.resize(kMaxTrackedPasses);
        m_scratch.reserve(kFrameHistoryCapacity);
        m_passOrder.reserve(kMaxTrackedPasses);
        m_passFilter.assign(64, '\0');
        m_textScratch.reserve(256);

        // Overview and Async Compute open by default: they answer the two
        // questions this window exists for. Everything else starts collapsed,
        // which also means its per-frame work is skipped until asked for.
        m_sectionOpenMask =
            (1u << static_cast<uint32_t>(Section::Overview)) |
            (1u << static_cast<uint32_t>(Section::AsyncCompute));
    }

    void RendererDiagnostics::sample(const FrameSample& frame)
    {
        // Freezing stops history from advancing so a spike can be read at
        // leisure. It affects only this observer; scheduling uses raw samples.
        // its own raw samples straight from the renderer and keeps running.
        if (m_paused || !m_windowOpen || !m_windowVisible)
        {
            return;
        }

        const float dt = std::max(0.0f, frame.deltaSeconds);
        const bool timingSectionOpen =
            sectionOpen(Section::Overview) ||
            sectionOpen(Section::AsyncCompute) ||
            sectionOpen(Section::GpuQueues);
        if (timingSectionOpen)
        {
            m_frameMs.push(frame.frameMs, dt);

            if (frame.timeline.valid)
            {
                const float graphics =
                    static_cast<float>(frame.timeline.graphicsBusyMs);
                const float compute =
                    static_cast<float>(frame.timeline.computeBusyMs);
                const float overlap =
                    static_cast<float>(frame.timeline.overlapMs);
                m_gpuGraphicsMs.push(graphics, dt);
                m_gpuComputeMs.push(compute, dt);
                m_gpuOverlapMs.push(overlap, dt);
                m_gpuBusyMs.push(graphics + compute - overlap, dt);
            }
        }

        // Per-pass series are keyed by node id and reused across frames; a pass
        // that did not run this frame simply gets no sample, which is why the
        // table reports its own sample count.
        if (sectionOpen(Section::PassTimings))
        {
            for (PassSeries& pass : m_passes)
            {
                pass.active = false;
            }
            for (const GpuPassSample& sample : frame.passSamples)
            {
                if (sample.node >= m_passes.size())
                {
                    continue;
                }
                PassSeries& pass = m_passes[sample.node];
                pass.node = sample.node;
                pass.queue = sample.queue;
                pass.active = true;
                pass.series.push(
                    static_cast<float>(sample.endMs - sample.beginMs), dt);
            }
        }

        m_displayAccumulator += dt;
        const float interval = 1.0f / std::clamp(m_displayHz, 4.0f, 10.0f);
        if (m_displayAccumulator >= interval)
        {
            m_displayAccumulator = 0.0f;
            m_displayDirty = true;
        }
    }

    void RendererDiagnostics::resetHistory()
    {
        m_frameMs.clear();
        m_gpuGraphicsMs.clear();
        m_gpuComputeMs.clear();
        m_gpuOverlapMs.clear();
        m_gpuBusyMs.clear();
        for (PassSeries& pass : m_passes)
        {
            pass.series.clear();
        }
        m_frameStats = {};
        m_gpuGraphicsStats = {};
        m_gpuComputeStats = {};
        m_gpuOverlapStats = {};
        m_gpuBusyStats = {};
        std::fill(m_passStats.begin(), m_passStats.end(),
            DiagnosticSeries::Stats{});
        m_displayDirty = true;
    }

    bool RendererDiagnostics::sectionOpen(Section section) const
    {
        return (m_sectionOpenMask & (1u << static_cast<uint32_t>(section))) != 0;
    }

    bool RendererDiagnostics::beginSection(Section section, const char* label)
    {
        const uint32_t bit = 1u << static_cast<uint32_t>(section);
        ImGui::SetNextItemOpen((m_sectionOpenMask & bit) != 0, ImGuiCond_Once);
        const bool open = ImGui::CollapsingHeader(label);
        if (open)
        {
            // A section that just opened has no statistics yet, because they
            // are only computed for visible sections. Ask for a refresh rather
            // than showing zeroes until the next display tick -- which, while
            // frozen, would never come.
            if ((m_sectionOpenMask & bit) == 0)
            {
                m_displayDirty = true;
                if (section == Section::PassTimings)
                {
                    for (PassSeries& pass : m_passes)
                    {
                        pass.active = false;
                    }
                }
            }
            m_sectionOpenMask |= bit;
        }
        else
        {
            m_sectionOpenMask &= ~bit;
        }
        return open;
    }

    void RendererDiagnostics::refreshDisplayStats()
    {
        // Only recomputed at the display rate, and only for what is visible:
        // each stats() call walks the window and runs an nth_element.
        if (sectionOpen(Section::Overview) || sectionOpen(Section::GpuQueues) ||
            sectionOpen(Section::AsyncCompute))
        {
            m_frameStats = m_frameMs.stats(m_historySeconds, m_scratch);
            m_gpuGraphicsStats =
                m_gpuGraphicsMs.stats(m_historySeconds, m_scratch);
            m_gpuComputeStats =
                m_gpuComputeMs.stats(m_historySeconds, m_scratch);
            m_gpuOverlapStats =
                m_gpuOverlapMs.stats(m_historySeconds, m_scratch);
            m_gpuBusyStats = m_gpuBusyMs.stats(m_historySeconds, m_scratch);
        }

        if (sectionOpen(Section::PassTimings))
        {
            for (size_t i = 0; i < m_passes.size(); ++i)
            {
                m_passStats[i] = m_passes[i].series.empty()
                    ? DiagnosticSeries::Stats{}
                    : m_passes[i].series.stats(m_historySeconds, m_scratch);
            }
        }

        m_displayDirty = false;
    }

    void RendererDiagnostics::helpMarker(const char* text)
    {
        // Contextual help belongs to the item that introduced the concept.
        // Keeping the label itself hoverable avoids a second noisy widget in
        // every row while preserving discoverability via delayed hover.
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
        {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 28.0f);
            ImGui::TextUnformatted(text);
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    }

    void RendererDiagnostics::drawSeriesRow(
        const char* label,
        const DiagnosticSeries::Stats& stats,
        const char* unit,
        const char* tooltip)
    {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(label);
        if (tooltip && tooltip[0] != '\0')
        {
            helpMarker(tooltip);
        }

        if (!stats.valid)
        {
            for (int i = 0; i < 5; ++i)
            {
                ImGui::TableNextColumn();
                ImGui::TextDisabled("-");
            }
            return;
        }

        ImGui::TableNextColumn();
        ImGui::Text("%.3f %s", stats.current, unit);
        ImGui::TableNextColumn();
        ImGui::Text("%.3f", stats.average);
        ImGui::TableNextColumn();
        ImGui::Text("%.3f", stats.minimum);
        ImGui::TableNextColumn();
        ImGui::Text("%.3f", stats.maximum);
        ImGui::TableNextColumn();
        ImGui::Text("%.3f", stats.p95);
    }

    void RendererDiagnostics::drawControls()
    {
        if (ImGui::Button(m_paused ? "Resume" : "Freeze"))
        {
            m_paused = !m_paused;
        }
        helpMarker(
            "Freezes this window's history and displayed values so a spike can "
            "be inspected. Rendering, and the async-compute policy's own "
            "measurements, are unaffected -- the policy reads raw per-frame "
            "samples directly and never sees anything shown here.");

        ImGui::SameLine();
        if (ImGui::Button("Reset"))
        {
            resetHistory();
        }
        helpMarker("Clears all accumulated history and statistics.");

        ImGui::SameLine();
        ImGui::SetNextItemWidth(120.0f);
        if (ImGui::SliderFloat("Refresh", &m_displayHz, 4.0f, 10.0f, "%.0f Hz"))
        {
            m_displayHz = std::clamp(m_displayHz, 4.0f, 10.0f);
        }
        helpMarker(
            "How often the displayed numbers update. Values are sampled every "
            "frame regardless; this only controls how fast the text changes, so "
            "that rapidly moving figures stay readable.");

        ImGui::SameLine();
        ImGui::SetNextItemWidth(140.0f);
        ImGui::SliderFloat(
            "History", &m_historySeconds, 0.25f, 8.0f, "%.2f s");
        helpMarker(
            "Time window that average / min / max / p95 are taken over. History "
            "is a fixed-size ring, so long windows saturate at the number of "
            "frames retained rather than growing memory.");

        if (m_paused)
        {
            ImGui::SameLine();
            ImGui::TextColored(kAmber, "FROZEN");
        }
    }

    void RendererDiagnostics::draw(Renderer& renderer)
    {
        if (!ImGui::GetCurrentContext() || !m_windowOpen)
        {
            m_windowVisible = false;
            return;
        }

        ImGui::SetNextWindowSize(ImVec2(660.0f, 620.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowPos(ImVec2(12.0f, 12.0f), ImGuiCond_FirstUseEver);
        ImVec4 background = ImGui::GetStyleColorVec4(ImGuiCol_WindowBg);
        background.w = 1.0f;
        ImGui::PushStyleColor(ImGuiCol_WindowBg, background);
        const bool contentsVisible =
            ImGui::Begin("Renderer Diagnostics", &m_windowOpen);
        ImGui::PopStyleColor();
        m_windowVisible = contentsVisible && m_windowOpen;
        if (!contentsVisible)
        {
            // Collapsed or clipped: ImGui has already skipped the contents, and
            // no statistics were recomputed.
            ImGui::End();
            return;
        }

        drawControls();
        ImGui::Separator();

        // Safe to run while frozen: pausing stops history from advancing, so
        // recomputing yields the same frozen numbers. It is what lets a section
        // opened during a freeze still populate.
        if (m_displayDirty)
        {
            refreshDisplayStats();
        }

        drawOverview(renderer);
        drawAsyncCompute(renderer);
        drawGpuQueues(renderer);
        drawFrameGraph(renderer);
        drawPassTimings(renderer);
        drawResources(renderer);
        drawVisibility(renderer);
        drawBackend(renderer);

        ImGui::End();
    }

    void RendererDiagnostics::drawOverview(Renderer& renderer)
    {
        if (!beginSection(Section::Overview, "Overview"))
        {
            return;
        }

        const float frameMs = m_frameStats.average;
        ImGui::Text(
            "%.1f FPS  |  %.3f ms",
            frameMs > 0.0f ? 1000.0f / frameMs : 0.0f,
            frameMs);
        helpMarker(
            "Averaged over the history window, not the last frame, so it does "
            "not flicker.");

        const RenderExtent extent = renderer.renderExtent();
        const BackendDiagnosticInfo backend = renderer.backendDiagnostics();
        ImGui::Text(
            "%s  |  %s  |  %ux%u",
            backend.backendName[0] ? backend.backendName : "Backend",
            renderPathName(renderer.renderPathType()),
            extent.width,
            extent.height);

        bool vsync = renderer.vsyncEnabled();
        if (ImGui::Checkbox("VSync", &vsync))
        {
            renderer.setVsyncEnabled(vsync);
        }
        helpMarker(
            "With VSync on the frame rate is capped by the monitor the window "
            "is on, so timing numbers describe the cap rather than the "
            "renderer's capability.");

        if (ImGui::BeginTable(
                "OverviewTimings", 6,
                ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                    ImGuiTableFlags_SizingStretchProp))
        {
            ImGui::TableSetupColumn("Metric");
            ImGui::TableSetupColumn("Current");
            ImGui::TableSetupColumn("Avg");
            ImGui::TableSetupColumn("Min");
            ImGui::TableSetupColumn("Max");
            ImGui::TableSetupColumn("p95");
            ImGui::TableHeadersRow();

            drawSeriesRow(
                "Frame (CPU wall)", m_frameStats, "ms",
                "Wall-clock time to produce a frame on the CPU thread, "
                "including any wait for the GPU. This is what FPS is derived "
                "from.");
            drawSeriesRow(
                "GPU busy", m_gpuBusyStats, "ms",
                "Time the GPU had work executing on any queue this frame "
                "(union of the graphics and compute timelines, so concurrent "
                "work is not double counted). Much lower than frame time means "
                "the GPU is idle waiting on the CPU.");
            ImGui::EndTable();
        }

        // The single most useful derived number here: it explains most
        // scheduling surprises, including why async compute may not help.
        if (m_frameStats.valid && m_gpuBusyStats.valid &&
            m_frameStats.average > 0.0f)
        {
            const float utilization =
                100.0f * m_gpuBusyStats.average / m_frameStats.average;
            ImGui::Text("GPU utilization: %.0f%%", utilization);
            ImGui::SameLine();
            if (utilization < 85.0f)
            {
                ImGui::TextColored(
                    kAmber, "(CPU-bound: GPU idle %.0f%% of the frame)",
                    100.0f - utilization);
            }
            else
            {
                ImGui::TextColored(kGreen, "(GPU-bound)");
            }
            helpMarker(
                "GPU busy time as a share of the frame. When this is low the "
                "frame is limited by the CPU, and making GPU work shorter or "
                "better overlapped cannot raise the frame rate.");
        }
        ImGui::Spacing();
    }

    void RendererDiagnostics::drawAsyncCompute(Renderer& renderer)
    {
        if (!beginSection(Section::AsyncCompute, "Async Compute"))
        {
            return;
        }

        const bool supported = renderer.backendSupportsAsyncCompute();
        const bool active = renderer.asyncComputeEnabled();

        bool requested = active;
        ImGui::BeginDisabled(!supported);
        if (ImGui::Checkbox("Enabled", &requested))
        {
            renderer.setAsyncComputeEnabled(requested);
        }
        ImGui::EndDisabled();
        helpMarker(
            "Enabled by default when the backend has a usable compute queue. "
            "This is a deterministic developer control; changing it performs "
            "one fence-safe frame-graph rebuild and is never overridden.");

        ImGui::SameLine();
        if (!supported)
        {
            ImGui::TextDisabled("(backend has no async queue)");
        }
        else if (active)
        {
            ImGui::TextColored(kGreen, "ACTIVE");
        }
        else
        {
            ImGui::TextColored(kAmber, "GRAPHICS FALLBACK");
        }

        if (ImGui::BeginTable(
                "AsyncTimings", 6,
                ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                    ImGuiTableFlags_SizingStretchProp))
        {
            ImGui::TableSetupColumn("Metric");
            ImGui::TableSetupColumn("Current");
            ImGui::TableSetupColumn("Avg");
            ImGui::TableSetupColumn("Min");
            ImGui::TableSetupColumn("Max");
            ImGui::TableSetupColumn("p95");
            ImGui::TableHeadersRow();
            drawSeriesRow(
                "Measured overlap", m_gpuOverlapStats, "ms",
                "Time the graphics and compute queues BOTH had work executing, "
                "measured from GPU timestamps. This is the only proof that "
                "async compute did anything: a pass can sit on the compute "
                "queue and still run entirely after graphics, which costs a "
                "submission and buys nothing.");
            drawSeriesRow(
                "Compute queue busy", m_gpuComputeStats, "ms",
                "Time the compute queue had work executing.");
            ImGui::EndTable();
        }

        if (m_gpuComputeStats.valid && m_gpuComputeStats.average > 0.0f)
        {
            const float hidden = 100.0f *
                m_gpuOverlapStats.average / m_gpuComputeStats.average;
            ImGui::Text("Compute hidden behind graphics: %.0f%%", hidden);
            helpMarker(
                "Share of compute-queue work that ran concurrently with "
                "graphics. 100% means it was entirely free; 0% means it was "
                "fully exposed and strictly worse than staying on graphics.");
        }

        // Submission cost: the other half of the trade the policy is making.
        const FrameGraphTopology topology = renderer.frameGraphTopology();
        ImGui::Text(
            "Submissions: %u batches (%u compute)  |  fence signals: %u  |  "
            "cross-queue waits: %u",
            topology.batches,
            topology.computeBatches,
            topology.batches,
            topology.crossQueueWaits);
        helpMarker(
            "The compiler emits one batch per (execution level, queue). Every "
            "batch costs a submit and a fence signal on the CPU, and each "
            "cross-queue edge adds a wait. This is what async compute must earn "
            "back through overlap.");
        ImGui::Spacing();
    }

    void RendererDiagnostics::drawGpuQueues(Renderer& renderer)
    {
        if (!beginSection(Section::GpuQueues, "GPU Queues"))
        {
            return;
        }

        if (ImGui::BeginTable(
                "QueueTimings", 6,
                ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                    ImGuiTableFlags_SizingStretchProp))
        {
            ImGui::TableSetupColumn("Queue");
            ImGui::TableSetupColumn("Current");
            ImGui::TableSetupColumn("Avg");
            ImGui::TableSetupColumn("Min");
            ImGui::TableSetupColumn("Max");
            ImGui::TableSetupColumn("p95");
            ImGui::TableHeadersRow();
            drawSeriesRow(
                "Graphics busy", m_gpuGraphicsStats, "ms",
                "Union of the graphics queue's pass intervals.");
            drawSeriesRow(
                "Compute busy", m_gpuComputeStats, "ms",
                "Union of the compute queue's pass intervals.");
            drawSeriesRow(
                "Overlap", m_gpuOverlapStats, "ms",
                "Intersection of the two: both queues executing at once.");
            ImGui::EndTable();
        }

        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextDisabled(
            "Transfer queue: not timed (copy queues need a separate timestamp "
            "heap type and take no part in the async-compute decision).");
        ImGui::PopTextWrapPos();

        // Per-pass timeline. Drawn from the last frame's raw samples, laid out
        // in the shared CPU-millisecond domain both queues are calibrated into.
        const std::span<const GpuPassSample> samples = renderer.gpuPassSamples();
        if (samples.empty())
        {
            ImGui::TextDisabled("No GPU samples resolved yet.");
            ImGui::Spacing();
            return;
        }

        double origin = samples[0].beginMs;
        double end = samples[0].endMs;
        for (const GpuPassSample& sample : samples)
        {
            origin = std::min(origin, sample.beginMs);
            end = std::max(end, sample.endMs);
        }
        const double span = std::max(1e-6, end - origin);

        ImGui::Text("Frame timeline (%.3f ms span)", span);
        helpMarker(
            "Each bar is one pass, positioned by its measured GPU start and "
            "end. Bars on different rows that overlap horizontally really did "
            "execute at the same time.");

        const float width = std::max(120.0f, ImGui::GetContentRegionAvail().x);
        const float rowHeight = ImGui::GetTextLineHeightWithSpacing();
        const ImVec2 base = ImGui::GetCursorScreenPos();
        ImDrawList* draw = ImGui::GetWindowDrawList();

        const CompiledGraphPlan& plan = renderer.compiledGraphPlan();
        for (uint32_t row = 0; row < 2; ++row)
        {
            const QueueType queue =
                row == 0 ? QueueType::Graphics : QueueType::Compute;
            const float y = base.y + static_cast<float>(row) * rowHeight;

            draw->AddText(ImVec2(base.x, y), ImGui::GetColorU32(ImGuiCol_Text),
                row == 0 ? "GFX" : "CMP");

            const float trackX = base.x + 34.0f;
            const float trackWidth = std::max(1.0f, width - 40.0f);
            draw->AddRectFilled(
                ImVec2(trackX, y + 1.0f),
                ImVec2(trackX + trackWidth, y + rowHeight - 3.0f),
                ImGui::GetColorU32(ImGuiCol_FrameBg));

            for (const GpuPassSample& sample : samples)
            {
                if (sample.queue != queue)
                {
                    continue;
                }
                const float x0 = trackX + trackWidth *
                    static_cast<float>((sample.beginMs - origin) / span);
                const float x1 = trackX + trackWidth *
                    static_cast<float>((sample.endMs - origin) / span);
                const ImU32 color = ImGui::GetColorU32(
                    queue == QueueType::Compute ? kOrange : kBlue);
                draw->AddRectFilled(
                    ImVec2(x0, y + 1.0f),
                    ImVec2(std::max(x1, x0 + 1.0f), y + rowHeight - 3.0f),
                    color);

                // Hover for identification: bars are often only pixels wide.
                if (ImGui::IsMouseHoveringRect(
                        ImVec2(x0, y + 1.0f),
                        ImVec2(std::max(x1, x0 + 3.0f), y + rowHeight - 3.0f)))
                {
                    ImGui::BeginTooltip();
                    ImGui::Text(
                        "%s\n%s | %.4f ms",
                        renderer.passName(sample.node),
                        queueName(sample.queue),
                        sample.endMs - sample.beginMs);
                    ImGui::EndTooltip();
                }
            }
        }
        ImGui::Dummy(ImVec2(width, rowHeight * 2.0f + 4.0f));
        (void)plan;
        ImGui::Spacing();
    }

    void RendererDiagnostics::drawFrameGraph(Renderer& renderer)
    {
        if (!beginSection(Section::FrameGraph, "Frame Graph"))
        {
            return;
        }

        const CompiledGraphPlan& plan = renderer.compiledGraphPlan();
        const FrameGraphTopology topology = renderer.frameGraphTopology();

        ImGui::Text(
            "Passes %u  |  Levels %u  |  Batches %u  |  Cross-queue waits %u",
            topology.passes, topology.levels, topology.batches,
            topology.crossQueueWaits);
        helpMarker(
            "Passes with no dependency between them share an execution level "
            "and are recorded in parallel on the CPU. Each (level, queue) pair "
            "becomes one submission batch.");

        // Which levels have batches on more than one queue. This is structure,
        // not evidence -- it says two queues were given work in the same level,
        // not that they were busy simultaneously. Measured overlap is in the
        // GPU Queues section.
        //
        // assign() into a member reuses the existing allocation; the graph only
        // changes on a rebuild, so this never grows in steady state.
        std::vector<uint32_t>& queuesPerLevel = m_levelQueueCounts;
        queuesPerLevel.assign(plan.executionLevels.size(), 0u);
        for (const QueueSubmissionBatch& batch : plan.queueSubmissions)
        {
            if (batch.levelIndex < queuesPerLevel.size())
            {
                ++queuesPerLevel[batch.levelIndex];
            }
        }

        if (!ImGui::BeginTable(
                "GraphSchedule", 8,
                ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                    ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp,
                ImVec2(0.0f, 260.0f)))
        {
            ImGui::Spacing();
            return;
        }

        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Lvl", ImGuiTableColumnFlags_WidthFixed, 32.0f);
        ImGui::TableSetupColumn("Queue");
        ImGui::TableSetupColumn("Pass");
        ImGui::TableSetupColumn("Cadence");
        ImGui::TableSetupColumn("Scheduling");
        ImGui::TableSetupColumn("Waits");
        ImGui::TableSetupColumn("Barriers");
        ImGui::TableSetupColumn("Node", ImGuiTableColumnFlags_WidthFixed, 40.0f);
        ImGui::TableHeadersRow();

        for (const QueueSubmissionBatch& batch : plan.queueSubmissions)
        {
            for (uint32_t i = 0; i < batch.nodeCount; ++i)
            {
                const GraphNodeId node =
                    plan.queueSubmissionNodes[batch.firstNode + i];
                if (node >= plan.nodes.size())
                {
                    continue;
                }
                const ExecutionNode& execution = plan.nodes[node];

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%u", batch.levelIndex);

                ImGui::TableNextColumn();
                if (batch.queue == QueueType::Compute)
                    ImGui::TextColored(kOrange, "Compute");
                else if (batch.queue == QueueType::Transfer)
                    ImGui::TextColored(kBlue, "Transfer");
                else
                    ImGui::TextUnformatted("Graphics");

                ImGui::TableNextColumn();
                ImGui::TextUnformatted(renderer.passName(node));

                ImGui::TableNextColumn();
                const bool scheduled = renderer.passScheduledThisFrame(node);
                switch (renderer.passCadence(node))
                {
                case PassCadence::Once:
                    scheduled
                        ? ImGui::TextColored(kBlue, "Once")
                        : ImGui::TextDisabled("Once (idle)");
                    break;
                case PassCadence::OnResize:
                    scheduled
                        ? ImGui::TextColored(kAmber, "On Resize")
                        : ImGui::TextDisabled("On Resize (idle)");
                    break;
                case PassCadence::OnInvalidation:
                    scheduled
                        ? ImGui::TextColored(kAmber, "Invalidated")
                        : ImGui::TextDisabled("Invalidated (idle)");
                    break;
                case PassCadence::PerFrame:
                default:
                    ImGui::TextDisabled("Per-Frame");
                    break;
                }

                // Eligibility vs actual scheduling: the distinction the async
                // policy turns on, so it is spelled out rather than merged.
                ImGui::TableNextColumn();
                const bool eligible = execution.asyncEligible;
                const bool onCompute = batch.queue == QueueType::Compute;
                const bool levelShared =
                    batch.levelIndex < queuesPerLevel.size() &&
                    queuesPerLevel[batch.levelIndex] > 1;
                if (onCompute && levelShared)
                    ImGui::TextColored(kGreen, "Async (shares level)");
                else if (onCompute)
                    ImGui::TextColored(kAmber, "Async (alone in level)");
                else if (eligible)
                    ImGui::TextColored(
                        ImVec4(0.7f, 0.7f, 0.5f, 1.0f), "Eligible, on graphics");
                else
                    ImGui::TextDisabled("-");

                ImGui::TableNextColumn();
                if (batch.waitCount == 0)
                {
                    ImGui::TextDisabled("-");
                }
                else
                {
                    m_textScratch.clear();
                    for (uint32_t wait = 0; wait < batch.waitCount; ++wait)
                    {
                        const uint32_t sourceIndex =
                            plan.queueSubmissionWaits[
                                batch.firstWait + wait].submissionIndex;
                        if (sourceIndex >= plan.queueSubmissions.size())
                        {
                            continue;
                        }
                        if (!m_textScratch.empty())
                        {
                            m_textScratch += "; ";
                        }
                        const QueueSubmissionBatch& source =
                            plan.queueSubmissions[sourceIndex];
                        m_textScratch += queueName(source.queue);
                        m_textScratch += ": ";
                        if (source.nodeCount == 0)
                        {
                            m_textScratch += "submission ";
                            m_textScratch += std::to_string(sourceIndex);
                        }
                        else
                        {
                            m_textScratch += renderer.passName(
                                plan.queueSubmissionNodes[source.firstNode]);
                            if (source.nodeCount > 1)
                            {
                                m_textScratch += " +";
                                m_textScratch +=
                                    std::to_string(source.nodeCount - 1);
                            }
                        }
                    }
                    ImGui::TextUnformatted(m_textScratch.c_str());
                }

                ImGui::TableNextColumn();
                if (node < plan.nodeSchedules.size())
                {
                    const NodeSchedule& schedule = plan.nodeSchedules[node];
                    ImGui::Text(
                        "%u in / %u out",
                        schedule.incomingBarrierCount,
                        schedule.outgoingBarrierCount);
                }
                else
                {
                    ImGui::TextDisabled("-");
                }

                ImGui::TableNextColumn();
                ImGui::Text("%u", node);
            }
        }
        ImGui::EndTable();

        ImGui::TextDisabled(
            "Dependencies: %u  |  Barriers: %u  |  Cross-frame edges: %u",
            static_cast<uint32_t>(plan.dependencies.size()),
            static_cast<uint32_t>(plan.barriers.size()),
            static_cast<uint32_t>(plan.crossFrameDependencies.size()));
        ImGui::Spacing();
    }

    void RendererDiagnostics::drawPassTimings(Renderer& renderer)
    {
        if (!beginSection(Section::PassTimings, "Pass Timings"))
        {
            return;
        }

        ImGui::SetNextItemWidth(200.0f);
        ImGui::InputText(
            "Filter", m_passFilter.data(), m_passFilter.size());
        ImGui::SameLine();
        ImGui::TextDisabled("(substring, case-insensitive)");

        static const char* kSortNames[] = { "Pass", "Queue", "Avg", "p95" };
        ImGui::SetNextItemWidth(120.0f);
        ImGui::Combo("Sort", &m_passSortColumn, kSortNames,
            IM_ARRAYSIZE(kSortNames));
        ImGui::SameLine();
        ImGui::Checkbox("Descending", &m_passSortDescending);

        // Built into preallocated member storage: clear() keeps the capacity,
        // so the table costs no allocations per frame.
        std::vector<uint32_t>& order = m_passOrder;
        order.clear();
        const char* filter = m_passFilter.data();
        const bool hasFilter = filter[0] != '\0';
        for (uint32_t i = 0; i < m_passes.size(); ++i)
        {
            if (m_passes[i].series.empty())
            {
                continue;
            }
            if (hasFilter)
            {
                const char* name = renderer.passName(m_passes[i].node);
                if (!name)
                {
                    continue;
                }
                // Case-insensitive substring test without allocating.
                bool match = false;
                for (const char* start = name; *start && !match; ++start)
                {
                    const char* a = start;
                    const char* b = filter;
                    while (*a && *b &&
                        std::tolower(static_cast<unsigned char>(*a)) ==
                            std::tolower(static_cast<unsigned char>(*b)))
                    {
                        ++a;
                        ++b;
                    }
                    match = (*b == '\0');
                }
                if (!match)
                {
                    continue;
                }
            }
            order.push_back(i);
        }

        std::sort(
            order.begin(), order.end(),
            [&](uint32_t a, uint32_t b)
            {
                int comparison = 0;
                switch (m_passSortColumn)
                {
                case 0:
                    comparison = std::strcmp(
                        renderer.passName(m_passes[a].node),
                        renderer.passName(m_passes[b].node));
                    break;
                case 1:
                    comparison = static_cast<int>(m_passes[a].queue) -
                        static_cast<int>(m_passes[b].queue);
                    break;
                case 3:
                    comparison = m_passStats[a].p95 < m_passStats[b].p95
                        ? -1 : (m_passStats[a].p95 > m_passStats[b].p95 ? 1 : 0);
                    break;
                case 2:
                default:
                    comparison = m_passStats[a].average < m_passStats[b].average
                        ? -1 : (m_passStats[a].average > m_passStats[b].average ? 1 : 0);
                    break;
                }
                if (comparison == 0)
                {
                    comparison = a < b ? -1 : (a > b ? 1 : 0);
                }
                return m_passSortDescending ? comparison > 0 : comparison < 0;
            });

        if (ImGui::BeginTable(
                "PassTimings", 6,
                ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                    ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp,
                ImVec2(0.0f, 240.0f)))
        {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("Pass");
            ImGui::TableSetupColumn("Queue");
            ImGui::TableSetupColumn("Current");
            ImGui::TableSetupColumn("Avg");
            ImGui::TableSetupColumn("p95");
            ImGui::TableSetupColumn("Max");
            ImGui::TableHeadersRow();

            float totalAverage = 0.0f;
            for (const uint32_t index : order)
            {
                const PassSeries& pass = m_passes[index];
                const DiagnosticSeries::Stats& stats = m_passStats[index];
                totalAverage += stats.average;

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                if (!pass.active)
                {
                    // Ran recently enough to still have history, but not this
                    // frame (a cadence-gated pass).
                    ImGui::TextDisabled("%s", renderer.passName(pass.node));
                }
                else
                {
                    ImGui::TextUnformatted(renderer.passName(pass.node));
                }
                ImGui::TableNextColumn();
                if (pass.queue == QueueType::Compute)
                    ImGui::TextColored(kOrange, "Compute");
                else
                    ImGui::TextUnformatted(queueName(pass.queue));
                ImGui::TableNextColumn();
                ImGui::Text("%.4f", stats.current);
                ImGui::TableNextColumn();
                ImGui::Text("%.4f", stats.average);
                ImGui::TableNextColumn();
                ImGui::Text("%.4f", stats.p95);
                ImGui::TableNextColumn();
                ImGui::Text("%.4f", stats.maximum);
            }
            ImGui::EndTable();

            ImGui::Text("Sum of pass averages: %.3f ms", totalAverage);
            helpMarker(
                "Sum, not wall time: passes that overlap on different queues "
                "are counted once each here, so this exceeds GPU busy time "
                "whenever async compute is active.");
        }
        ImGui::Spacing();
    }

    void RendererDiagnostics::drawResources(Renderer& renderer)
    {
        if (!beginSection(Section::Resources, "Resources"))
        {
            return;
        }

        const CompiledGraphPlan& plan = renderer.compiledGraphPlan();

        uint32_t transient = 0;
        uint32_t persistent = 0;
        uint32_t imported = 0;
        uint32_t history = 0;
        uint64_t totalBytes = 0;
        for (const GraphResource& resource : plan.resources)
        {
            switch (resource.ownership)
            {
            case ResourceOwnership::Transient:  ++transient; break;
            case ResourceOwnership::Persistent: ++persistent; break;
            case ResourceOwnership::Imported:   ++imported; break;
            }
            if (resource.multiplicity == ResourceMultiplicity::History)
            {
                ++history;
            }
            if (resource.type == GraphResourceType::Buffer)
            {
                totalBytes += resource.bufferDesc.size;
            }
        }

        char bytes[32];
        formatBytes(totalBytes, bytes, sizeof(bytes));
        ImGui::Text(
            "Resources %u  |  transient %u, persistent %u, imported %u, "
            "history %u  |  declared buffer bytes %s",
            static_cast<uint32_t>(plan.resources.size()),
            transient, persistent, imported, history, bytes);

        ImGui::TextDisabled(
            "Aliasing: none. Transient resources are each backed by their own "
            "allocation; the frame graph does not currently alias memory "
            "between resources with disjoint lifetimes. The First/Last columns "
            "below are what an aliasing pass would key off.");
        helpMarker(
            "Lifetimes are expressed as indices into the execution order: a "
            "resource is live from its first accessing pass to its last. Two "
            "transient resources whose ranges do not intersect could share "
            "memory if aliasing were implemented.");

        if (!ImGui::BeginTable(
                "Resources", 7,
                ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                    ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp,
                ImVec2(0.0f, 240.0f)))
        {
            ImGui::Spacing();
            return;
        }

        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Resource");
        ImGui::TableSetupColumn("Type");
        ImGui::TableSetupColumn("Ownership");
        ImGui::TableSetupColumn("Instances");
        ImGui::TableSetupColumn("Size");
        ImGui::TableSetupColumn("First");
        ImGui::TableSetupColumn("Last");
        ImGui::TableHeadersRow();

        for (const ResourceLifetime& lifetime : plan.resourceLifetimes)
        {
            if (lifetime.resource >= plan.resources.size())
            {
                continue;
            }
            const GraphResource& resource = plan.resources[lifetime.resource];
            const bool isTexture = resource.type == GraphResourceType::Texture;

            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            const char* name = isTexture
                ? resource.textureDesc.debugName
                : resource.bufferDesc.debugName;
            ImGui::TextUnformatted(name && name[0] ? name : "(unnamed)");

            ImGui::TableNextColumn();
            ImGui::TextUnformatted(isTexture ? "Texture" : "Buffer");

            ImGui::TableNextColumn();
            ImGui::TextUnformatted(ownershipName(resource.ownership));

            ImGui::TableNextColumn();
            ImGui::TextUnformatted(multiplicityName(resource.multiplicity));

            ImGui::TableNextColumn();
            if (isTexture)
            {
                ImGui::Text(
                    "%ux%u m%u",
                    resource.textureDesc.width,
                    resource.textureDesc.height,
                    resource.textureDesc.mipLevels);
            }
            else
            {
                formatBytes(resource.bufferDesc.size, bytes, sizeof(bytes));
                ImGui::TextUnformatted(bytes);
            }

            ImGui::TableNextColumn();
            ImGui::Text("%u", lifetime.firstUse);
            ImGui::TableNextColumn();
            ImGui::Text("%u", lifetime.lastUse);
        }
        ImGui::EndTable();
        ImGui::Spacing();
    }

    void RendererDiagnostics::drawVisibility(Renderer& renderer)
    {
        if (!beginSection(Section::Visibility, "Visibility / Culling"))
        {
            return;
        }

        const RenderPathType path = renderer.renderPathType();
        if (path != RenderPathType::Forward &&
            path != RenderPathType::ClusteredForward)
        {
            ImGui::TextDisabled(
                "GPU culling runs only on the forward paths.");
            ImGui::Spacing();
            return;
        }

        bool occlusion = renderer.gpuOcclusionEnabled();
        if (ImGui::Checkbox("GPU occlusion culling", &occlusion))
        {
            renderer.setGpuOcclusionEnabled(occlusion);
        }
        helpMarker(
            "Tests instances against the previous frame's Hi-Z pyramid. "
            "Disabling it leaves frustum culling in place.");

        int debugMode = static_cast<int>(renderer.gpuCullDebugMode());
        static const char* kDebugModes[] = {
            "Off", "Statistics", "Classification View" };
        ImGui::SetNextItemWidth(180.0f);
        if (ImGui::Combo(
                "Cull debug", &debugMode, kDebugModes,
                IM_ARRAYSIZE(kDebugModes)))
        {
            renderer.setGpuCullDebugMode(
                static_cast<GpuCullDebugMode>(std::clamp(debugMode, 0, 2)));
        }
        helpMarker(
            "Statistics reads back per-frame cull counts. Classification View "
            "additionally tints each instance by the decision made about it. "
            "Both add a readback and a validation dispatch, so leave this Off "
            "when measuring.");

        if (path == RenderPathType::ClusteredForward)
        {
            bool heatmap = renderer.clusteredForwardHeatmapEnabled();
            if (ImGui::Checkbox("Cluster light heatmap", &heatmap))
            {
                renderer.setClusteredForwardHeatmapEnabled(heatmap);
            }
            helpMarker("Tints pixels by the number of lights in their cluster.");

            const GlobalIlluminationQuality currentQuality =
                renderer.globalIlluminationQuality();
            const auto& currentPreset =
                globalIlluminationPresetInfo(currentQuality);
            ImGui::SetNextItemWidth(210.0f);
            if (ImGui::BeginCombo("RT / diffuse GI quality",
                    currentQuality == GlobalIlluminationQuality::Custom
                        ? "Custom" : currentPreset.name.data()))
            {
                for (const auto& preset : globalIlluminationPresets())
                {
                    const bool selected = currentQuality == preset.quality;
                    if (ImGui::Selectable(preset.name.data(), selected))
                        renderer.setGlobalIlluminationQuality(preset.quality);
                    if (selected) ImGui::SetItemDefaultFocus();
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("%.*s\nTarget %.2f ms; memory limit %.0f MiB",
                            static_cast<int>(preset.description.size()),
                            preset.description.data(),
                            preset.gpuTimeTargetMilliseconds,
                            static_cast<double>(preset.memoryLimitBytes) /
                                (1024.0 * 1024.0));
                }
                if (ImGui::Selectable("Custom",
                        currentQuality == GlobalIlluminationQuality::Custom))
                    renderer.setGlobalIlluminationQuality(
                        GlobalIlluminationQuality::Custom);
                ImGui::EndCombo();
            }
            helpMarker(
                "Presets change coherent spatial, ray, update, and reconstruction "
                "budgets without changing lighting intensity. Off removes GI and "
                "its hardware-RT demand when no other RT path is active.");

            if (currentQuality == GlobalIlluminationQuality::Custom)
            {
                auto custom = renderer.globalIlluminationConfiguration();
                bool changed = false;
                int clipmaps = static_cast<int>(custom.limits.probeClipmapCount);
                int resolution = static_cast<int>(custom.limits.probeResolution);
                int updates = static_cast<int>(custom.limits.maxProbeUpdates);
                int rays = static_cast<int>(custom.limits.rayBudget);
                int visibilityRays = std::clamp(
                    rays / std::max(updates, 1), 1, 32);
                int divisor = static_cast<int>(custom.evaluationDivisor);
                int memoryMiB = static_cast<int>(custom.memoryLimitBytes /
                    (1024ull * 1024ull));
                float gpuTarget = custom.gpuTimeTargetMilliseconds;
                ImGui::SetNextItemWidth(130.0f);
                changed |= ImGui::SliderInt("Probe clipmaps", &clipmaps, 1, 8);
                ImGui::SetNextItemWidth(130.0f);
                changed |= ImGui::SliderInt("Probe resolution", &resolution, 8, 64);
                ImGui::SetNextItemWidth(130.0f);
                changed |= ImGui::InputInt("Probe updates / frame", &updates, 64, 256);
                ImGui::SetNextItemWidth(130.0f);
                changed |= ImGui::InputInt("GI ray budget", &rays, 1024, 8192);
                ImGui::SetNextItemWidth(130.0f);
                if (ImGui::SliderInt(
                        "Visibility rays / probe", &visibilityRays, 1, 32))
                {
                    rays = std::max(updates, 64) * visibilityRays;
                    changed = true;
                }
                ImGui::SetNextItemWidth(130.0f);
                changed |= ImGui::SliderInt(
                    "Temporal reconstruction divisor", &divisor, 2, 4);
                ImGui::SetNextItemWidth(130.0f);
                changed |= ImGui::InputInt(
                    "GI memory limit (MiB)", &memoryMiB, 16, 64);
                ImGui::SetNextItemWidth(130.0f);
                changed |= ImGui::InputFloat(
                    "GI GPU target (ms)", &gpuTarget, 0.1f, 0.5f, "%.2f");
                changed |= ImGui::Checkbox("Surfel detail", &custom.surfelDetail);
                if (changed)
                {
                    custom.quality = GlobalIlluminationQuality::Custom;
                    custom.limits.probeClipmapCount =
                        static_cast<uint32_t>(std::max(clipmaps, 1));
                    custom.limits.probeResolution =
                        static_cast<uint32_t>(std::max(resolution, 2));
                    custom.limits.maxProbeUpdates =
                        static_cast<uint32_t>(std::max(updates, 64));
                    custom.limits.rayBudget =
                        static_cast<uint32_t>(std::max(rays, 1));
                    custom.evaluationDivisor = divisor <= 2 ? 2u : 4u;
                    custom.memoryLimitBytes = static_cast<uint64_t>(
                        std::max(memoryMiB, 0)) * 1024ull * 1024ull;
                    custom.gpuTimeTargetMilliseconds =
                        std::max(gpuTarget, 0.0f);
                    renderer.setGlobalIlluminationConfiguration(custom);
                }
            }

            const auto views = globalIlluminationDebugViews();
            const GlobalIlluminationDebugView selectedView =
                renderer.globalIlluminationDebugView();
            const auto& selectedViewInfo =
                globalIlluminationDebugViewInfo(selectedView);
            ImGui::SetNextItemWidth(250.0f);
            if (ImGui::BeginCombo("GI debug", selectedViewInfo.name.data()))
            {
                for (const auto& view : views)
                {
                    const bool selected = selectedView == view.view;
                    if (ImGui::Selectable(view.name.data(), selected))
                        renderer.setGlobalIlluminationDebugView(view.view);
                    if (selected) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            if (ImGui::IsItemHovered())
            {
                ImGui::BeginTooltip();
                ImGui::TextWrapped("%.*s", static_cast<int>(selectedViewInfo.tooltip.size()),
                    selectedViewInfo.tooltip.data());
                ImGui::Separator();
                ImGui::Text("Legend: %.*s", static_cast<int>(selectedViewInfo.legend.size()),
                    selectedViewInfo.legend.data());
                ImGui::Text("Units/range: %.*s", static_cast<int>(selectedViewInfo.unitsAndRange.size()),
                    selectedViewInfo.unitsAndRange.data());
                ImGui::Text("Source: %.*s / %.*s",
                    static_cast<int>(selectedViewInfo.sourcePass.size()), selectedViewInfo.sourcePass.data(),
                    static_cast<int>(selectedViewInfo.sourceResource.size()), selectedViewInfo.sourceResource.data());
                ImGui::Text("Class: %s", selectedViewInfo.filtered
                    ? "cached / filtered (must be stable)" : "raw state (may change/noise)");
                ImGui::EndTooltip();
            }
            if (selectedViewInfo.radiometric)
            {
                float exposure = renderer.globalIlluminationDebugExposure();
                ImGui::SetNextItemWidth(210.0f);
                if (ImGui::SliderFloat("GI debug exposure", &exposure,
                        0.03125f, 32.0f, "%.3fx",
                        ImGuiSliderFlags_Logarithmic))
                    renderer.setGlobalIlluminationDebugExposure(exposure);
            }
            if (selectedView ==
                    GlobalIlluminationDebugView::DiagnosticIntensity)
            {
                float intensity =
                    renderer.globalIlluminationDiagnosticIntensity();
                ImGui::SetNextItemWidth(210.0f);
                if (ImGui::SliderFloat(
                        "GI diagnostic intensity", &intensity,
                        0.0f, 16.0f, "%.2fx"))
                {
                    renderer.setGlobalIlluminationDiagnosticIntensity(intensity);
                }
            }
            helpMarker(
                "The intensity slider affects only its diagnostic view. "
                "Normal shading remains fixed at physically neutral 1.0x.");

            const GlobalIlluminationRuntimeStatistics giStats =
                renderer.globalIlluminationStatistics();
            ImGui::Text("GI: %s | %.3f ms inclusive",
                giStats.active ? "active" :
                    (giStats.memoryLimitExceeded ? "memory limit exceeded" :
                        "fallback off"),
                giStats.inclusiveGpuMilliseconds);
            ImGui::Text("%u probe updates | %u rays | %.2f MiB (probes %.2f MiB)",
                giStats.configuredProbeUpdates, giStats.configuredRayBudget,
                static_cast<double>(giStats.allocatedBytes) / (1024.0 * 1024.0),
                static_cast<double>(giStats.probeBytes) / (1024.0 * 1024.0));
        }

        if (renderer.gpuCullDebugMode() != GpuCullDebugMode::Off)
        {
            const GpuCullStats stats = renderer.gpuCullStats();
            const float input = static_cast<float>(std::max(1u, stats.inputCount));
            ImGui::Separator();
            ImGui::Text(
                "Input %u  |  Visible %u", stats.inputCount, stats.visible);
            ImGui::TextColored(
                ImVec4(0.2f, 0.55f, 1.0f, 1.0f),
                "Frustum culled: %u (%.1f%%)",
                stats.frustumCulled, 100.0f * stats.frustumCulled / input);
            ImGui::TextColored(
                kRed, "Occlusion culled: %u (%.1f%%)",
                stats.occlusionCulled, 100.0f * stats.occlusionCulled / input);
            ImGui::TextColored(
                kAmber, "Conservative retained: %u (%.1f%%)",
                stats.conservativeRetained,
                100.0f * stats.conservativeRetained / input);
            helpMarker(
                "Kept because the Hi-Z test could not prove they were hidden "
                "(e.g. no valid history this frame). Conservative means "
                "possibly-wasted work, never a missing object.");

            ImGui::Text(
                "Discrepancies: false-occluded %u | false-visible %u",
                stats.falseOccluded, stats.falseVisible);
            helpMarker(
                "From the validation dispatch, which re-tests the culling "
                "result. false-occluded means an instance was culled but should "
                "have been drawn -- a correctness bug, and it would show as "
                "popping. false-visible means it was drawn needlessly, which "
                "only costs performance.");
            if (stats.falseOccluded > 0)
            {
                ImGui::SameLine();
                ImGui::TextColored(kRed, "<- visible artifacts");
            }

            ImGui::Text(
                "Hi-Z history: %s  |  Overflow: %u",
                stats.historyValid ? "valid" : "fallback (first frame/resize)",
                stats.overflow);

            const GpuCullPerformance performance =
                renderer.gpuCullPerformance();
            ImGui::Text(
                "Cull cost: GPU %.3f ms | CPU record %.3f ms",
                performance.gpuMilliseconds,
                performance.cpuRecordMilliseconds);
            ImGui::TextDisabled(
                "Classification colors: green visible, blue frustum, red "
                "occlusion, yellow conservative.");
        }

        // The Hi-Z pyramid viewer, previously its own backend-owned window.
        ImGui::Separator();
        bool hiZView = renderer.hiZDebugViewEnabled();
        if (ImGui::Checkbox("Hi-Z pyramid view", &hiZView))
        {
            renderer.setHiZDebugViewEnabled(hiZView);
        }
        helpMarker(
            "The depth pyramid occlusion culling tests against. Each mip is a "
            "conservative (farthest) reduction of the one below it.");

        if (!hiZView)
        {
            ImGui::Spacing();
            return;
        }

        bool previous = renderer.hiZDebugPrevious();
        if (ImGui::Checkbox("Previous frame", &previous))
        {
            renderer.setHiZDebugPrevious(previous);
        }
        helpMarker(
            "Occlusion culling tests against the PREVIOUS frame's pyramid, so "
            "this is what the culling actually saw.");

        int mip = static_cast<int>(renderer.hiZDebugMip());
        ImGui::SetNextItemWidth(180.0f);
        if (ImGui::SliderInt("Mip", &mip, 0, 15))
        {
            renderer.setHiZDebugMip(static_cast<uint32_t>(std::max(0, mip)));
        }

        const HiZDebugImage image = renderer.hiZDebugImage(
            previous, static_cast<uint32_t>(std::max(0, mip)));
        if (!image.valid)
        {
            ImGui::TextDisabled("Waiting for Hi-Z pyramid...");
            ImGui::Spacing();
            return;
        }

        ImGui::Text(
            "%s | mip %d / %u | %ux%u",
            previous ? "Previous" : "Current",
            mip, image.mipLevels, image.width, image.height);

        const float available = std::max(64.0f, ImGui::GetContentRegionAvail().x);
        const float aspect = static_cast<float>(image.width) /
            static_cast<float>(std::max(1u, image.height));
        const float drawWidth = std::min(available, 512.0f);
        ImGui::Image(
            static_cast<ImTextureID>(image.textureId),
            ImVec2(drawWidth, drawWidth / std::max(0.01f, aspect)));
        ImGui::Spacing();
    }

    void RendererDiagnostics::drawBackend(Renderer& renderer)
    {
        if (!beginSection(Section::Backend, "Backend / Capabilities"))
        {
            return;
        }

        const BackendDiagnosticInfo info = renderer.backendDiagnostics();
        ImGui::Text(
            "%s  |  %s",
            info.backendName[0] ? info.backendName : "Backend",
            info.adapterName[0] ? info.adapterName : "Unknown adapter");
        ImGui::Text(
            "Async compute queue: %s",
            renderer.backendSupportsAsyncCompute() ? "yes" : "no");

        if (!info.features.empty() &&
            ImGui::BeginTable(
                "BackendFeatures", 3,
                ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                    ImGuiTableFlags_SizingStretchProp))
        {
            ImGui::TableSetupColumn("Feature");
            ImGui::TableSetupColumn("State");
            ImGui::TableSetupColumn("Notes");
            ImGui::TableHeadersRow();
            for (const BackendFeature& feature : info.features)
            {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(feature.name);
                ImGui::TableNextColumn();
                if (feature.enabled)
                    ImGui::TextColored(kGreen, "enabled");
                else
                    ImGui::TextDisabled("off");
                ImGui::TableNextColumn();
                ImGui::TextDisabled(
                    "%s", feature.detail && feature.detail[0]
                        ? feature.detail : "-");
            }
            ImGui::EndTable();
        }

        if (!info.limits.empty() &&
            ImGui::BeginTable(
                "BackendLimits", 2,
                ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                    ImGuiTableFlags_SizingStretchProp))
        {
            ImGui::TableSetupColumn("Limit");
            ImGui::TableSetupColumn("Value");
            ImGui::TableHeadersRow();
            for (const BackendLimit& limit : info.limits)
            {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(limit.name);
                ImGui::TableNextColumn();
                ImGui::Text("%llu",
                    static_cast<unsigned long long>(limit.value));
            }
            ImGui::EndTable();
        }
        ImGui::Spacing();
    }
}
