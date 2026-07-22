// ic/renderer/renderer.h
#pragma once
#include "ic/common/ic_common.h"
#include "gpu_queue_profiler.h"
#include "ibl_baker.h"
#include "renderer_backend.h"
#include "renderer_specification.h"
#include "ic/renderer/frame_graph/compiled_graph_plan.h"
#include "ic/renderer/frame_graph/frame_graph_pass.h"
#include "ic/renderer/renderer_path/renderer_path.h"
namespace ic
{
    struct FrameContext;
    struct RendererSpecification;
    struct SceneRenderView;

    class Window;
    class RendererBackend;
    class RendererPath;


    class Renderer
    {
    public:
        Renderer(
            const RendererSpecification& spec);

        ~Renderer();


        void init(
            RendererSpecification& spec,
            Window& window,
            uint32_t workerCount
        );
        void shutdown();

        void render(FrameContext& frame);
        void render(
            FrameContext& frame,
            const SceneRenderView& scene);
        bool beginDebugGuiFrame();
        void endDebugGuiFrame();
        [[nodiscard]] bool vsyncEnabled() const;
        void setVsyncEnabled(bool enabled);
        [[nodiscard]] bool clusteredForwardHeatmapEnabled() const;
        void setClusteredForwardHeatmapEnabled(bool enabled);
        [[nodiscard]] bool hiZDebugViewEnabled() const;
        void setHiZDebugViewEnabled(bool enabled);
        [[nodiscard]] uint32_t hiZDebugMip() const;
        void setHiZDebugMip(uint32_t mip);
        [[nodiscard]] bool hiZDebugPrevious() const;
        void setHiZDebugPrevious(bool previous);
        [[nodiscard]] bool gpuOcclusionEnabled() const;
        void setGpuOcclusionEnabled(bool enabled);
        [[nodiscard]] GpuCullDebugMode gpuCullDebugMode() const;
        void setGpuCullDebugMode(GpuCullDebugMode mode);
        [[nodiscard]] GpuCullStats gpuCullStats() const;
        [[nodiscard]] GpuCullPerformance gpuCullPerformance() const;
        // Measured per-queue GPU occupancy for the most recently completed
        // frame, from timestamps. This is what distinguishes async compute that
        // actually overlaps graphics from async compute that merely relocated
        // work onto another queue and added synchronization.
        [[nodiscard]] GpuQueueTimelineStats gpuQueueTimeline() const;
        // Raw per-pass GPU intervals behind gpuQueueTimeline(), for callers
        // that need a per-pass breakdown rather than the frame summary.
        [[nodiscard]] std::span<const GpuPassSample> gpuPassSamples() const;
        void setGpuProfilingEnabled(bool enabled);
        [[nodiscard]] bool gpuProfilingEnabled() const;
        // Benchmark/control hook: 0 closes diagnostics; otherwise bit N opens
        // diagnostics section N (Overview through Backend).
        void setDiagnosticsSectionMask(uint32_t mask);
        [[nodiscard]] const char* passName(GraphNodeId node) const;
        [[nodiscard]] FrameGraphTopology frameGraphTopology() const;

        // Read-only views for diagnostics. These expose what the renderer
        // already computed for its own use; nothing here drives scheduling.
        [[nodiscard]] const CompiledGraphPlan& compiledGraphPlan() const;
        [[nodiscard]] PassCadence passCadence(GraphNodeId node) const;
        // Whether this pass's cadence gate let it run this frame.
        [[nodiscard]] bool passScheduledThisFrame(GraphNodeId node) const;
        [[nodiscard]] RenderExtent renderExtent() const;
        [[nodiscard]] bool backendSupportsAsyncCompute() const;
        [[nodiscard]] BackendDiagnosticInfo backendDiagnostics() const;
        [[nodiscard]] RendererPerformanceCounters performanceCounters() const;
        [[nodiscard]] HiZDebugImage hiZDebugImage(bool previous, uint32_t mip);
        // Whether cross-queue async compute is currently active. This is the
        // effective value: the scheduler's decision AND backend support.
        // Changing it rebuilds the frame graph on the next frame so queue
        // assignments update.
        [[nodiscard]] bool asyncComputeEnabled() const;
        // Deterministic developer control. Enabled by default when supported;
        // changing it performs one fence-safe graph transition and is never
        // overridden by runtime measurements.
        void setAsyncComputeEnabled(bool enabled);
        [[nodiscard]] bool globalIlluminationEnabled() const;
        void setGlobalIlluminationEnabled(bool enabled);
        [[nodiscard]] GlobalIlluminationQuality globalIlluminationQuality() const;
        void setGlobalIlluminationQuality(GlobalIlluminationQuality quality);
        [[nodiscard]] GlobalIlluminationConfiguration
            globalIlluminationConfiguration() const;
        void setGlobalIlluminationConfiguration(
            const GlobalIlluminationConfiguration& configuration);
        [[nodiscard]] GlobalIlluminationDebugView globalIlluminationDebugView() const;
        void setGlobalIlluminationDebugView(GlobalIlluminationDebugView view);
        [[nodiscard]] float globalIlluminationDiagnosticIntensity() const;
        void setGlobalIlluminationDiagnosticIntensity(float intensity);
        [[nodiscard]] float globalIlluminationDebugExposure() const;
        void setGlobalIlluminationDebugExposure(float exposure);
        [[nodiscard]] bool rayTracingEnabled() const;
        void setRayTracingEnabled(bool enabled);
        [[nodiscard]] GlobalIlluminationRuntimeStatistics
            globalIlluminationStatistics() const;
        [[nodiscard]] RenderPathType renderPathType() const;
        
        void buildOrRebuildFrameGraph();
        // Explicitly dirties passes whose execution policy subscribes to one
        // or more of these API-neutral reasons. Manual is suitable for tools
        // and one-off editor actions.
        void invalidatePasses(PassInvalidation reasons);

        [[nodiscard]] IBLHandle requestIBLBake(
            const IBLBakeDesc& desc);
        [[nodiscard]] IBLBakeState iblState(
            IBLHandle handle) const;
        [[nodiscard]] IBLProbeSnapshot iblSnapshot(
            IBLHandle handle) const;

        static Scope<RendererBackend> createBackend(
            RendererBackendType type);

        static Scope<RendererPath> createPath(RenderPathType type);

    private:

        void syncSceneEnvironmentIBL(
            FrameContext& frame,
            const SceneRenderView& scene);

        void processPendingRendererJobs(
            FrameContext& frame);

        void updateGpuTimeline();
        void updateRayTracingDemand();
        void requestAsyncComputeRebuild();

        struct Runtime;
        Scope<Runtime> m_runtime;

    };
}
