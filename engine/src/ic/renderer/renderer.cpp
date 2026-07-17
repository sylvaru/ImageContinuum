// ic/renderer/renderer.h
#include "ic/common/ic_pch.h"
#include "ic/core/app_base.h"
#include "ic/core/frame_context.h"
#include "ic/renderer/renderer.h"
#include "ic/renderer/renderer_backend.h"
#include "ic/renderer/vulkan_backend/vulkan_backend.h"
#include "ic/renderer/renderer_path/renderer_path.h"
#include "ic/renderer/renderer_path/forward.h"
#include "ic/renderer/renderer_path/clustered_forward.h"
#include "ic/renderer/renderer_path/path_tracer.h"
#include "ic/renderer/frame_graph/frame_graph_builder.h"
#include "ic/renderer/frame_graph/frame_graph_compiler.h"
#include "ic/renderer/frame_graph/frame_graph_arena.h"
#include "ic/renderer/frame_graph/compiled_graph_plan.h"
#include "ic/renderer/pipeline_library.h"
#include "ic/renderer/renderer_diagnostics.h"
#include "ic/scene/scene_render_view.h"

#include <spdlog/spdlog.h>
#include <imgui.h>

#ifdef _WIN32
#include "ic/renderer/dx12_backend/dx12_backend.h"
#endif

namespace ic
{
    namespace
    {
        bool sameSceneEnvironmentBake(
            const IBLBakeDesc& lhs,
            const IBLBakeDesc& rhs)
        {
            return lhs.sourceEnvironment == rhs.sourceEnvironment &&
                   lhs.environmentSize == rhs.environmentSize &&
                   lhs.irradianceSize == rhs.irradianceSize &&
                   lhs.prefilterSize == rhs.prefilterSize &&
                   lhs.brdfLutSize == rhs.brdfLutSize &&
                   lhs.format == rhs.format;
        }

        PassExecutionPolicy executionPolicy(const PassPayload& payload)
        {
            if (const auto* pass = std::get_if<GraphicsPassData>(&payload))
                return pass->execution;
            if (const auto* pass = std::get_if<ComputePassData>(&payload))
                return pass->execution;
            if (const auto* pass = std::get_if<TransferPassData>(&payload))
                return pass->execution;
            return {};
        }
    }

    struct Renderer::Runtime
    {
        FrameGraphArena arena;

        CompiledGraphPlan compiledGraphPlan;

        Scope<RendererBackend> backend;
        Scope<RendererPath> path;
        PipelineLibrary pipelineLibrary;
        IBLBaker iblBaker;

        FrameGraphBuilder builder;
        FrameGraphCompiler compiler;

        IBLBakeDesc activeSceneEnvironmentBake = {};
        IBLHandle activeSceneEnvironmentIBL = {};
        bool hasActiveSceneEnvironmentBake = false;

        RenderExtent renderExtent{};
        // Last render-surface generation the graph was built against. A change
        // (swapchain recreation) forces a rebuild so the graph, swapchain and
        // materialized resources always share one extent.
        uint64_t lastSurfaceGeneration = 0;
        bool graphDirty = true;
        bool schedulingTransitionPending = false;
        // Deterministic: async is requested by default and falls back only when
        // the backend has no usable compute queue. The developer can turn it
        // off explicitly; no runtime trials mutate this value.
        bool asyncComputeRequested = true;
        // This frame's queue timeline, analyzed exactly once (in
        // updateGpuTimeline) and shared by diagnostics and benchmarks. The
        // analysis is an interval merge over
        // every pass sample, so it must not be repeated per reader.
        GpuQueueTimelineStats gpuTimeline{};
        std::vector<GpuPassSample> graphicsTimelineScratch;
        std::vector<GpuPassSample> computeTimelineScratch;
        // Observer only: reads what the renderer already computed and never
        // feeds anything back. Costs nothing while the debug GUI is off.
        RendererDiagnostics diagnostics;
        bool debugGuiEnabled = true;
        GpuCullDebugMode gpuCullDebugMode = GpuCullDebugMode::Off;
        FrameGraphBuildReason rebuildReason =
            FrameGraphBuildReason::Startup;
        RenderPathType pathType = RenderPathType::Forward;
        PassInvalidation pendingInvalidation =
            PassInvalidation::Startup | PassInvalidation::GraphRebuild;
        std::vector<uint8_t> nodeExecuted;
        std::vector<uint8_t> executeNodes;
        uint64_t lastSceneVersion = UINT64_MAX;
        uint64_t lastEnvironmentVersion = UINT64_MAX;
        float lastNearPlane = 0.0f;
        float lastFarPlane = 0.0f;
        float lastVerticalFov = 0.0f;
        float lastAspectRatio = 0.0f;

        Runtime(
            Scope<RendererBackend> b,
            Scope<RendererPath> p)
            : backend(std::move(b))
            , path(std::move(p))
            , builder(arena.resource())
            , compiler(arena.resource())
        {
            graphicsTimelineScratch.reserve(128);
            computeTimelineScratch.reserve(128);
        }
    };

    Renderer::Renderer(
        const RendererSpecification& spec)
    {
        m_runtime = std::make_unique<Runtime>(
            createBackend(spec.backendType),
            createPath(spec.pathType)
        );
        m_runtime->pathType = spec.pathType;
    }
    Renderer::~Renderer()
    {}

    void Renderer::init(
        RendererSpecification& spec,
        Window& window,
        uint32_t workerCount)
    {
        spdlog::info("[Renderer] init...");

        if (!spec.pipelineLibraryPath.empty())
        {
            m_runtime->pipelineLibrary.load(spec.pipelineLibraryPath);
        }

        m_runtime->backend->init(
            spec,
            m_runtime->pipelineLibrary,
            window,
            workerCount);
        m_runtime->backend->setGpuOcclusionEnabled(
            spec.settings.gpuOcclusion);
        m_runtime->gpuCullDebugMode =
            spec.settings.gpuCullDebugMode;
        m_runtime->backend->setGpuCullDebugMode(
            spec.settings.gpuCullDebugMode);
        m_runtime->debugGuiEnabled = spec.useDebugGui;

        m_runtime->renderExtent = {
            std::max(1u, window.getWidth()),
            std::max(1u, window.getHeight())            
        };

        buildOrRebuildFrameGraph();
    }

    void Renderer::render(FrameContext& frame)
    {
        static const SceneRenderView emptyScene{};
        render(frame, emptyScene);
    }

    void Renderer::render(
        FrameContext& frame,
        [[maybe_unused]] const SceneRenderView& scene)
    {
        auto& runtime = *m_runtime;

        if (!runtime.backend)
        {
            return;
        }

        // Reconcile the swapchain to the window and adopt its framebuffer extent
        // as the single authoritative render extent for this frame. This runs
        // before the graph is (re)built, so the compiled graph and every
        // materialized attachment match the backbuffer. No swapchain recreation
        // happens later in the frame to desynchronize them. A non-renderable
        // surface (minimized / zero area) skips the whole frame; because
        // materialize is never reached, the per-slot/history rings do not
        // advance, so a skipped frame cannot stale history.
        const RenderSurfaceState surface =
            runtime.backend->reconcileRenderSurface();
        if (!surface.renderable)
        {
            return;
        }

        const RenderExtent authoritativeExtent{
            std::max(1u, surface.width),
            std::max(1u, surface.height) };
        if (authoritativeExtent.width != runtime.renderExtent.width ||
            authoritativeExtent.height != runtime.renderExtent.height ||
            surface.generation != runtime.lastSurfaceGeneration)
        {
            runtime.renderExtent = authoritativeExtent;
            runtime.lastSurfaceGeneration = surface.generation;
            runtime.graphDirty = true;
            runtime.rebuildReason = FrameGraphBuildReason::Resize;
            runtime.pendingInvalidation |= PassInvalidation::Resize;
        }

        // Runs before the rebuild check so a decision change this frame is
        // applied immediately rather than a frame late. It reads the previous
        // frame's resolved timestamps, so it never stalls on the GPU. This also
        // caches runtime.gpuTimeline for the frame.
        updateGpuTimeline();

        // Diagnostics observe the SAME raw values the policy just consumed.
        // Skipped entirely when the debug GUI is off, and never read back into
        // the renderer.
        if (runtime.debugGuiEnabled)
        {
            runtime.diagnostics.sample({
                .deltaSeconds = frame.deltaTime,
                .frameMs = frame.deltaTime * 1000.0f,
                .timeline = runtime.gpuTimeline,
                .passSamples = runtime.backend->gpuPassSamples()
            });
        }

        if (runtime.graphDirty)
        {
            if (runtime.schedulingTransitionPending)
            {
                runtime.backend->drainForSchedulingTransition();
                runtime.schedulingTransitionPending = false;
            }
            buildOrRebuildFrameGraph();
        }

        syncSceneEnvironmentIBL(frame, scene);
        processPendingRendererJobs(frame);

        if (runtime.lastSceneVersion != scene.sceneVersion)
        {
            runtime.lastSceneVersion = scene.sceneVersion;
            runtime.pendingInvalidation |= PassInvalidation::Scene;
        }
        if (runtime.lastEnvironmentVersion != scene.environment.version)
        {
            runtime.lastEnvironmentVersion = scene.environment.version;
            runtime.pendingInvalidation |= PassInvalidation::Environment;
        }
        if (runtime.lastNearPlane != scene.camera.nearPlane ||
            runtime.lastFarPlane != scene.camera.farPlane ||
            runtime.lastVerticalFov != scene.camera.verticalFovRadians ||
            runtime.lastAspectRatio != scene.camera.aspectRatio)
        {
            runtime.lastNearPlane = scene.camera.nearPlane;
            runtime.lastFarPlane = scene.camera.farPlane;
            runtime.lastVerticalFov = scene.camera.verticalFovRadians;
            runtime.lastAspectRatio = scene.camera.aspectRatio;
            runtime.pendingInvalidation |= PassInvalidation::Configuration;
        }

        if (scene.camera.valid == 0u)
        {
            return;
        }
        if (frame.services && frame.services->assetManager)
        {
            for (const SceneModelRenderItem& item : scene.models)
            {
                const AssetState state =
                    frame.services->assetManager->state(item.model);
                if (state != AssetState::Loaded && state != AssetState::Failed)
                {
                    return;
                }
            }
        }

        const CompiledGraphPlan& plan = runtime.compiledGraphPlan;
        runtime.executeNodes.assign(plan.nodes.size(), uint8_t{1});
        for (const ExecutionNode& node : plan.nodes)
        {
            PassExecutionPolicy policy{};
            if (node.payloadIndex < plan.payloads.size())
            {
                policy = executionPolicy(plan.payloads[node.payloadIndex]);
            }

            bool execute = true;
            if (policy.cadence == PassCadence::Once)
            {
                execute = node.nodeId >= runtime.nodeExecuted.size() ||
                    runtime.nodeExecuted[node.nodeId] == 0;
            }
            else if (policy.cadence == PassCadence::OnResize)
            {
                execute = node.nodeId >= runtime.nodeExecuted.size() ||
                    runtime.nodeExecuted[node.nodeId] == 0 ||
                    any(runtime.pendingInvalidation, PassInvalidation::Resize);
            }
            else if (policy.cadence == PassCadence::OnInvalidation)
            {
                execute = node.nodeId >= runtime.nodeExecuted.size() ||
                    runtime.nodeExecuted[node.nodeId] == 0 ||
                    any(runtime.pendingInvalidation, policy.invalidation);
            }

            // A skipped producer needs storage that survives the frame. Writes
            // to transient or imported resources are therefore always active.
            if (!execute)
            {
                const auto accesses = plan.resourceAccesses.subspan(
                    node.firstResourceAccess, node.resourceAccessCount);
                for (const ResourceAccess& access : accesses)
                {
                    if (access.access != AccessType::Read &&
                        access.resource < plan.resources.size() &&
                        plan.resources[access.resource].ownership !=
                            ResourceOwnership::Persistent)
                    {
                        execute = true;
                        break;
                    }
                }
            }
            runtime.executeNodes[node.nodeId] = execute ? 1u : 0u;
        }

        const GraphExecutionContext execution{ runtime.executeNodes };
        const bool completed = runtime.backend->execute(
            plan,
            execution,
            frame,
            scene);

        if (completed)
        {
            for (GraphNodeId node = 0; node < runtime.executeNodes.size(); ++node)
            {
                if (runtime.executeNodes[node]) runtime.nodeExecuted[node] = 1;
            }
            runtime.pendingInvalidation = PassInvalidation::None;
        }
    }

    void Renderer::shutdown()
    {
        if (m_runtime && m_runtime->backend)
        {
            m_runtime->backend->shutdown();
        }
    }

    void Renderer::buildOrRebuildFrameGraph()
    {
        auto& rt = *m_runtime;

        rt.builder.clear();

        RendererPathContext rctx;
        rctx.renderExtent = m_runtime->renderExtent;
        rctx.rebuildReason = rt.rebuildReason;
        rctx.asyncComputeEnabled =
            rt.asyncComputeRequested &&
            rt.backend && rt.backend->supportsAsyncCompute();
        rctx.occlusionDiagnosticsEnabled =
            rt.gpuCullDebugMode != GpuCullDebugMode::Off;

        rt.path->buildFrameGraph(rctx, rt.builder);

        rt.compiledGraphPlan = rt.compiler.compile(rt.builder);

        rt.nodeExecuted.assign(rt.compiledGraphPlan.nodes.size(), uint8_t{0});
        rt.executeNodes.assign(rt.compiledGraphPlan.nodes.size(), uint8_t{1});
        rt.pendingInvalidation |= PassInvalidation::GraphRebuild;

        rt.graphDirty = false;
        rt.rebuildReason = FrameGraphBuildReason::Explicit;
    }

    void Renderer::invalidatePasses(PassInvalidation reasons)
    {
        if (m_runtime)
        {
            m_runtime->pendingInvalidation |= reasons;
        }
    }

    Scope<RendererBackend> 
        Renderer::createBackend(
            RendererBackendType type)
    {
        switch (type)
        {
        case RendererBackendType::Vulkan:
            return std::make_unique<VulkanBackend>();

#ifdef _WIN32
        case RendererBackendType::DX12:
            return std::make_unique<DX12Backend>();

#endif
        default:
            return nullptr;
        }
    }

    Scope<RendererPath> 
        Renderer::createPath(RenderPathType type)
    {
        switch (type)
        {
        case RenderPathType::Forward:
            return std::make_unique<ForwardRendererPath>();

        case RenderPathType::ClusteredForward:
            return std::make_unique<ClusteredForwardRendererPath>();

        case RenderPathType::PathTraced:
            return std::make_unique<PathTracerRendererPath>();

        }

        return nullptr;
    }

    bool Renderer::beginDebugGuiFrame()
    {
        return m_runtime &&
            m_runtime->backend &&
            m_runtime->backend->beginDebugGuiFrame();
    }

    void Renderer::endDebugGuiFrame()
    {
        if (m_runtime && m_runtime->backend)
        {
            // One consolidated window; the backends no longer own popups of
            // their own (the Hi-Z pyramid is drawn inside it via
            // RendererBackend::hiZDebugImage).
            m_runtime->diagnostics.draw(*this);
            m_runtime->backend->endDebugGuiFrame();
        }
    }

    bool Renderer::vsyncEnabled() const
    {
        return m_runtime &&
            m_runtime->backend &&
            m_runtime->backend->vsyncEnabled();
    }

    void Renderer::setVsyncEnabled(bool enabled)
    {
        if (m_runtime && m_runtime->backend)
        {
            m_runtime->backend->setVsyncEnabled(enabled);
        }
    }

    bool Renderer::clusteredForwardHeatmapEnabled() const
    {
        return m_runtime &&
            m_runtime->backend &&
            m_runtime->backend->clusteredForwardHeatmapEnabled();
    }

    void Renderer::setClusteredForwardHeatmapEnabled(bool enabled)
    {
        if (m_runtime && m_runtime->backend)
        {
            m_runtime->backend->setClusteredForwardHeatmapEnabled(enabled);
        }
    }

    bool Renderer::hiZDebugViewEnabled() const
    {
        return m_runtime &&
            m_runtime->backend &&
            m_runtime->backend->hiZDebugViewEnabled();
    }

    void Renderer::setHiZDebugViewEnabled(bool enabled)
    {
        if (m_runtime && m_runtime->backend)
        {
            m_runtime->backend->setHiZDebugViewEnabled(enabled);
        }
    }

    uint32_t Renderer::hiZDebugMip() const
    {
        return m_runtime && m_runtime->backend
            ? m_runtime->backend->hiZDebugMip()
            : 0u;
    }

    void Renderer::setHiZDebugMip(uint32_t mip)
    {
        if (m_runtime && m_runtime->backend)
        {
            m_runtime->backend->setHiZDebugMip(mip);
        }
    }

    bool Renderer::hiZDebugPrevious() const
    {
        return m_runtime && m_runtime->backend &&
            m_runtime->backend->hiZDebugPrevious();
    }

    void Renderer::setHiZDebugPrevious(bool previous)
    {
        if (m_runtime && m_runtime->backend)
        {
            m_runtime->backend->setHiZDebugPrevious(previous);
        }
    }

    bool Renderer::gpuOcclusionEnabled() const
    {
        return m_runtime && m_runtime->backend &&
            m_runtime->backend->gpuOcclusionEnabled();
    }

    void Renderer::setGpuOcclusionEnabled(bool enabled)
    {
        if (m_runtime && m_runtime->backend)
        {
            m_runtime->backend->setGpuOcclusionEnabled(enabled);
        }
    }

    GpuCullDebugMode Renderer::gpuCullDebugMode() const
    {
        return m_runtime ? m_runtime->gpuCullDebugMode
            : GpuCullDebugMode::Off;
    }

    void Renderer::setGpuCullDebugMode(GpuCullDebugMode mode)
    {
        if (!m_runtime || m_runtime->gpuCullDebugMode == mode)
        {
            return;
        }
        m_runtime->gpuCullDebugMode = mode;
        if (m_runtime->backend)
        {
            m_runtime->backend->setGpuCullDebugMode(mode);
        }
        m_runtime->graphDirty = true;
        m_runtime->rebuildReason = FrameGraphBuildReason::Explicit;
        m_runtime->pendingInvalidation |= PassInvalidation::Configuration;
    }

    GpuCullStats Renderer::gpuCullStats() const
    {
        return m_runtime && m_runtime->backend
            ? m_runtime->backend->gpuCullStats()
            : GpuCullStats{};
    }

    GpuCullPerformance Renderer::gpuCullPerformance() const
    {
        return m_runtime && m_runtime->backend
            ? m_runtime->backend->gpuCullPerformance()
            : GpuCullPerformance{};
    }

    const char* Renderer::passName(GraphNodeId node) const
    {
        if (!m_runtime)
        {
            return "Invalid";
        }

        const CompiledGraphPlan& plan = m_runtime->compiledGraphPlan;
        if (node >= plan.nodes.size())
        {
            return "Invalid";
        }
        const ExecutionNode& execution = plan.nodes[node];
        if (execution.payloadIndex >= plan.payloads.size())
        {
            return "Unnamed";
        }

        const PassPayload& payload = plan.payloads[execution.payloadIndex];
        if (const auto* pass = std::get_if<GraphicsPassData>(&payload))
            return pass->name.c_str();
        if (const auto* pass = std::get_if<ComputePassData>(&payload))
            return pass->name.c_str();
        if (const auto* pass = std::get_if<TransferPassData>(&payload))
            return pass->name.c_str();
        if (std::holds_alternative<EnvironmentConvertPassData>(payload))
            return "EnvironmentConvert";
        if (std::holds_alternative<PathTracePassData>(payload))
            return "PathTrace";
        if (std::holds_alternative<TonemapPassData>(payload))
            return "Tonemap";
        if (std::holds_alternative<PresentPassData>(payload))
            return "Present";
        return "Unnamed";
    }

    std::span<const GpuPassSample> Renderer::gpuPassSamples() const
    {
        return m_runtime && m_runtime->backend
            ? m_runtime->backend->gpuPassSamples()
            : std::span<const GpuPassSample>{};
    }

    const CompiledGraphPlan& Renderer::compiledGraphPlan() const
    {
        static const CompiledGraphPlan kEmpty{};
        return m_runtime ? m_runtime->compiledGraphPlan : kEmpty;
    }

    PassCadence Renderer::passCadence(GraphNodeId node) const
    {
        if (!m_runtime)
        {
            return PassCadence::PerFrame;
        }

        const CompiledGraphPlan& plan = m_runtime->compiledGraphPlan;
        if (node >= plan.nodes.size())
        {
            return PassCadence::PerFrame;
        }
        const ExecutionNode& execution = plan.nodes[node];
        if (execution.payloadIndex >= plan.payloads.size())
        {
            return PassCadence::PerFrame;
        }

        // Only Graphics/Compute/Transfer passes carry an author-declared
        // cadence; anything else is implicitly per-frame.
        const PassPayload& payload = plan.payloads[execution.payloadIndex];
        if (const auto* pass = std::get_if<GraphicsPassData>(&payload))
            return pass->execution.cadence;
        if (const auto* pass = std::get_if<ComputePassData>(&payload))
            return pass->execution.cadence;
        if (const auto* pass = std::get_if<TransferPassData>(&payload))
            return pass->execution.cadence;
        return PassCadence::PerFrame;
    }

    bool Renderer::passScheduledThisFrame(GraphNodeId node) const
    {
        return m_runtime &&
            node < m_runtime->executeNodes.size() &&
            m_runtime->executeNodes[node] != 0;
    }

    RenderExtent Renderer::renderExtent() const
    {
        return m_runtime ? m_runtime->renderExtent : RenderExtent{};
    }

    bool Renderer::backendSupportsAsyncCompute() const
    {
        return m_runtime && m_runtime->backend &&
            m_runtime->backend->supportsAsyncCompute();
    }

    BackendDiagnosticInfo Renderer::backendDiagnostics() const
    {
        return m_runtime && m_runtime->backend
            ? m_runtime->backend->backendDiagnostics()
            : BackendDiagnosticInfo{};
    }

    HiZDebugImage Renderer::hiZDebugImage(bool previous, uint32_t mip)
    {
        return m_runtime && m_runtime->backend
            ? m_runtime->backend->hiZDebugImage(previous, mip)
            : HiZDebugImage{};
    }

    FrameGraphTopology Renderer::frameGraphTopology() const
    {
        FrameGraphTopology topology{};
        if (!m_runtime)
        {
            return topology;
        }

        const CompiledGraphPlan& plan = m_runtime->compiledGraphPlan;
        topology.passes = static_cast<uint32_t>(plan.nodes.size());
        topology.levels = static_cast<uint32_t>(plan.executionLevels.size());
        topology.batches = static_cast<uint32_t>(plan.queueSubmissions.size());
        topology.crossQueueWaits =
            static_cast<uint32_t>(plan.queueSubmissionWaits.size());
        for (const QueueSubmissionBatch& batch : plan.queueSubmissions)
        {
            topology.computeBatches +=
                batch.queue == QueueType::Compute ? 1u : 0u;
        }
        return topology;
    }

    GpuQueueTimelineStats Renderer::gpuQueueTimeline() const
    {
        // Cached by updateGpuTimeline earlier this frame rather than
        // re-analyzed per caller.
        return m_runtime ? m_runtime->gpuTimeline : GpuQueueTimelineStats{};
    }

    RendererPerformanceCounters Renderer::performanceCounters() const
    {
        return m_runtime && m_runtime->backend
            ? m_runtime->backend->performanceCounters()
            : RendererPerformanceCounters{};
    }

    void Renderer::setGpuProfilingEnabled(bool enabled)
    {
        if (m_runtime && m_runtime->backend)
        {
            m_runtime->backend->setGpuProfilingEnabled(enabled);
        }
    }

    bool Renderer::gpuProfilingEnabled() const
    {
        return m_runtime && m_runtime->backend &&
            m_runtime->backend->gpuProfilingEnabled();
    }

    void Renderer::setDiagnosticsSectionMask(uint32_t mask)
    {
        if (!m_runtime)
        {
            return;
        }
        m_runtime->diagnostics.setWindowOpen(mask != 0u);
        m_runtime->diagnostics.setSectionOpenMask(mask);
    }

    bool Renderer::asyncComputeEnabled() const
    {
        return m_runtime &&
            m_runtime->asyncComputeRequested &&
            m_runtime->backend &&
            m_runtime->backend->supportsAsyncCompute();
    }

    void Renderer::setAsyncComputeEnabled(bool enabled)
    {
        if (!m_runtime || m_runtime->asyncComputeRequested == enabled)
        {
            return;
        }
        const bool wasEnabled = asyncComputeEnabled();
        m_runtime->asyncComputeRequested = enabled;
        if (asyncComputeEnabled() != wasEnabled)
        {
            requestAsyncComputeRebuild();
        }
    }

    void Renderer::requestAsyncComputeRebuild()
    {
        // Queue assignment is baked at graph-build time. Rebuild on the next
        // frame outside recording, after a one-shot all-queue drain; waiting a
        // single reused frame slot is insufficient when older submissions may
        // still use resources and cross-frame edges from the previous plan.
        m_runtime->schedulingTransitionPending = true;
        m_runtime->graphDirty = true;
        m_runtime->rebuildReason = FrameGraphBuildReason::Explicit;
        m_runtime->pendingInvalidation |= PassInvalidation::GraphRebuild;
    }

    void Renderer::updateGpuTimeline()
    {
        auto& rt = *m_runtime;

        const std::span<const GpuPassSample> samples =
            rt.backend ? rt.backend->gpuPassSamples()
                       : std::span<const GpuPassSample>{};
        // Analyzed once per frame and cached for every other reader.
        rt.gpuTimeline = analyzeGpuQueueTimeline(
            samples,
            rt.graphicsTimelineScratch,
            rt.computeTimelineScratch);
    }

    RenderPathType Renderer::renderPathType() const
    {
        return m_runtime ? m_runtime->pathType : RenderPathType::Forward;
    }

    IBLHandle Renderer::requestIBLBake(
        const IBLBakeDesc& desc)
    {
        if (!m_runtime)
        {
            return {};
        }

        return m_runtime->iblBaker.requestBake(desc);
    }

    IBLBakeState Renderer::iblState(
        IBLHandle handle) const
    {
        if (!m_runtime)
        {
            return IBLBakeState::Empty;
        }

        return m_runtime->iblBaker.state(handle);
    }

    IBLProbeSnapshot Renderer::iblSnapshot(
        IBLHandle handle) const
    {
        if (!m_runtime)
        {
            return {};
        }

        return m_runtime->iblBaker.snapshot(handle);
    }

    void Renderer::syncSceneEnvironmentIBL(
        FrameContext& frame,
        const SceneRenderView& scene)
    {
        auto& rt = *m_runtime;

        if (scene.environment.enabled && scene.environment.equirectTexture)
        {
            IBLBakeDesc bakeDesc{};
            bakeDesc.sourceEnvironment =
                scene.environment.equirectTexture;
            bakeDesc.environmentSize =
                scene.environment.settings.cubemapSize;
            bakeDesc.debugName = "SceneEnvironment";

            const bool needsSceneEnvironmentBake =
                !rt.hasActiveSceneEnvironmentBake ||
                !sameSceneEnvironmentBake(
                    rt.activeSceneEnvironmentBake,
                    bakeDesc);

            if (needsSceneEnvironmentBake)
            {
                rt.activeSceneEnvironmentBake = bakeDesc;
                rt.activeSceneEnvironmentIBL =
                    rt.iblBaker.requestBake(bakeDesc);
                rt.hasActiveSceneEnvironmentBake =
                    static_cast<bool>(rt.activeSceneEnvironmentIBL);

                spdlog::info(
                    "[Renderer] Scene environment bake transition: "
                    "version={} source={}:{} envSize={} handle={}:{}",
                    scene.environment.version,
                    bakeDesc.sourceEnvironment.index,
                    bakeDesc.sourceEnvironment.generation,
                    bakeDesc.environmentSize,
                    rt.activeSceneEnvironmentIBL.index,
                    rt.activeSceneEnvironmentIBL.generation);
            }
            else if (
                rt.iblBaker.state(rt.activeSceneEnvironmentIBL) ==
                IBLBakeState::WaitingForSource)
            {
                AssetManager* assets =
                    frame.services ? frame.services->assetManager : nullptr;

                if (assets &&
                    assets->isLoaded(
                        rt.activeSceneEnvironmentBake.sourceEnvironment) &&
                    rt.iblBaker.requeue(rt.activeSceneEnvironmentIBL))
                {
                    spdlog::info(
                        "[Renderer] Scene environment source is ready; "
                        "requeueing IBL bake handle={}:{}",
                        rt.activeSceneEnvironmentIBL.index,
                        rt.activeSceneEnvironmentIBL.generation);
                }
            }

            return;
        }

        if (rt.hasActiveSceneEnvironmentBake)
        {
            spdlog::info(
                "[Renderer] Scene environment disabled or missing; "
                "clearing active IBL handle {}:{}",
                rt.activeSceneEnvironmentIBL.index,
                rt.activeSceneEnvironmentIBL.generation);

            rt.activeSceneEnvironmentBake = {};
            rt.activeSceneEnvironmentIBL = {};
            rt.hasActiveSceneEnvironmentBake = false;
        }
    }

    void Renderer::processPendingRendererJobs(FrameContext& frame)
    {
        auto& rt = *m_runtime;

        std::vector<IBLBakeRequest> iblRequests =
            rt.iblBaker.takePendingRequests();

        if (iblRequests.empty())
        {
            return;
        }

        spdlog::info(
            "[Renderer] Executing {} queued IBL bake request(s)",
            iblRequests.size());

        std::vector<IBLBakeResult> results =
            rt.backend->executeIBLBakeRequests(
                iblRequests,
                frame);

        spdlog::info(
            "[Renderer] IBL bake backend returned {} result(s)",
            results.size());

        for (const IBLBakeResult& result : results)
        {
            if (!result.success && result.error == "SourceNotReady")
            {
                rt.iblBaker.markWaitingForSource(
                    result.handle,
                    result.requestId);
            }
            else
            {
                rt.iblBaker.publishResult(result);
            }
        }
    }
}
