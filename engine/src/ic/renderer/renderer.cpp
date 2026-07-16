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
        // User-facing async-compute toggle. The effective state also requires
        // backend->supportsAsyncCompute(); see buildOrRebuildFrameGraph.
        bool asyncComputeEnabled = true;
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

        if (runtime.graphDirty)
        {
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
            rt.asyncComputeEnabled &&
            rt.backend && rt.backend->supportsAsyncCompute();

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
            drawFrameGraphDebugWindow();
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

    bool Renderer::asyncComputeEnabled() const
    {
        return m_runtime &&
            m_runtime->asyncComputeEnabled &&
            m_runtime->backend &&
            m_runtime->backend->supportsAsyncCompute();
    }

    void Renderer::setAsyncComputeEnabled(bool enabled)
    {
        if (!m_runtime || m_runtime->asyncComputeEnabled == enabled)
        {
            return;
        }
        m_runtime->asyncComputeEnabled = enabled;
        // Queue assignment is baked at graph-build time; rebuild on the next
        // frame (matching the resize path) so the change takes effect safely
        // outside GPU recording.
        m_runtime->graphDirty = true;
        m_runtime->rebuildReason = FrameGraphBuildReason::Explicit;
        m_runtime->pendingInvalidation |= PassInvalidation::GraphRebuild;
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

    void Renderer::drawFrameGraphDebugWindow()
    {
        if (!ImGui::GetCurrentContext() || !m_runtime)
        {
            return;
        }

        static bool open = true;
        if (!open)
        {
            return;
        }
        if (!ImGui::Begin("Frame Graph Queues", &open))
        {
            ImGui::End();
            return;
        }

        const CompiledGraphPlan& plan = m_runtime->compiledGraphPlan;

        // Per-level count of distinct queue submissions. The compiler emits at
        // most one batch per (level, queue), so a level with >1 batch is one
        // where passes on different queues actually execute concurrently on the
        // GPU (only re-synchronized by the queue fences). That is the ground
        // truth for "real overlap occurred", distinct from mere eligibility.
        std::vector<uint32_t> queuesPerLevel(plan.executionLevels.size(), 0u);
        for (const QueueSubmissionBatch& batch : plan.queueSubmissions)
        {
            if (batch.levelIndex < queuesPerLevel.size())
                ++queuesPerLevel[batch.levelIndex];
        }
        uint32_t overlappingLevels = 0;
        uint32_t computeBatches = 0;
        for (uint32_t count : queuesPerLevel)
            overlappingLevels += count > 1 ? 1u : 0u;
        for (const QueueSubmissionBatch& batch : plan.queueSubmissions)
            computeBatches += batch.queue == QueueType::Compute ? 1u : 0u;

        const float framerate = ImGui::GetIO().Framerate;
        ImGui::Text("%.1f FPS (%.2f ms) | %s", framerate,
            framerate > 0.0f ? 1000.0f / framerate : 0.0f,
            vsyncEnabled() ? "VSync on" : "VSync off");
        ImGui::Text("Passes %u | Levels %u | Batches %u",
            static_cast<uint32_t>(plan.nodes.size()),
            static_cast<uint32_t>(plan.executionLevels.size()),
            static_cast<uint32_t>(plan.queueSubmissions.size()));

        const bool backendAsync =
            m_runtime->backend && m_runtime->backend->supportsAsyncCompute();
        bool asyncToggle = m_runtime->asyncComputeEnabled;
        ImGui::BeginDisabled(!backendAsync);
        if (ImGui::Checkbox("Async compute", &asyncToggle))
        {
            setAsyncComputeEnabled(asyncToggle);
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (!backendAsync)
            ImGui::TextDisabled("(backend has no async queue)");
        else if (!asyncComputeEnabled())
            ImGui::TextDisabled("(graphics fallback)");
        else if (overlappingLevels > 0)
            ImGui::TextColored(ImVec4(0.25f, 0.9f, 0.35f, 1.0f),
                "(active: %u compute pass%s overlap graphics across %u level%s)",
                computeBatches, computeBatches == 1 ? "" : "es",
                overlappingLevels, overlappingLevels == 1 ? "" : "s");
        else
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.35f, 1.0f),
                "(on compute queue, but no level overlaps graphics)");

        ImGui::TextDisabled(
            "Queue = actual submit queue. Async/Overlap: Overlap = runs "
            "concurrently with another queue; Eligible = compute pass on the "
            "graphics queue (async off).");

        auto queueName = [](QueueType queue)
            {
                switch (queue)
                {
                case QueueType::Graphics: return "Graphics";
                case QueueType::Compute: return "Compute";
                case QueueType::Transfer: return "Transfer";
                }
                return "Unknown";
            };

        auto passName = [&plan](GraphNodeId node) -> const char*
            {
                if (node >= plan.nodes.size()) return "Invalid";
                const ExecutionNode& execution = plan.nodes[node];
                if (execution.payloadIndex >= plan.payloads.size()) return "Unnamed";
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
            };

        // Author-declared execution cadence (see PassCadence). Only Graphics,
        // Compute and Transfer passes carry it; anything else is treated as
        // per-frame.
        auto passCadence = [&plan](GraphNodeId node) -> PassCadence
            {
                if (node >= plan.nodes.size()) return PassCadence::PerFrame;
                const ExecutionNode& execution = plan.nodes[node];
                if (execution.payloadIndex >= plan.payloads.size())
                    return PassCadence::PerFrame;
                const PassPayload& payload = plan.payloads[execution.payloadIndex];
                if (const auto* pass = std::get_if<GraphicsPassData>(&payload))
                    return pass->execution.cadence;
                if (const auto* pass = std::get_if<ComputePassData>(&payload))
                    return pass->execution.cadence;
                if (const auto* pass = std::get_if<TransferPassData>(&payload))
                    return pass->execution.cadence;
                return PassCadence::PerFrame;
            };

        if (ImGui::BeginTable("GraphQueueSchedule", 8,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
        {
            ImGui::TableSetupColumn("Level");
            ImGui::TableSetupColumn("Queue");
            ImGui::TableSetupColumn("Pass");
            ImGui::TableSetupColumn("Cadence");
            ImGui::TableSetupColumn("Waits");
            ImGui::TableSetupColumn("CPU Parallel");
            ImGui::TableSetupColumn("Async/Overlap");
            ImGui::TableSetupColumn("Node");
            ImGui::TableHeadersRow();

            for (const QueueSubmissionBatch& batch : plan.queueSubmissions)
            {
                const bool cpuParallel =
                    batch.levelIndex < plan.executionLevels.size() &&
                    plan.executionLevels[batch.levelIndex].nodeCount > 1;
                for (uint32_t i = 0; i < batch.nodeCount; ++i)
                {
                    const GraphNodeId node =
                        plan.queueSubmissionNodes[batch.firstNode + i];
                    ImGui::TableNextRow();
                    const bool levelOverlaps =
                        batch.levelIndex < queuesPerLevel.size() &&
                        queuesPerLevel[batch.levelIndex] > 1;

                    ImGui::TableNextColumn();
                    ImGui::Text("%u", batch.levelIndex);
                    ImGui::TableNextColumn();
                    // Actual submit queue. Highlight non-graphics queues so it is
                    // obvious at a glance which passes left the graphics queue.
                    if (batch.queue == QueueType::Compute)
                        ImGui::TextColored(
                            ImVec4(1.0f, 0.65f, 0.25f, 1.0f), "Compute");
                    else if (batch.queue == QueueType::Transfer)
                        ImGui::TextColored(
                            ImVec4(0.6f, 0.7f, 1.0f, 1.0f), "Transfer");
                    else
                        ImGui::TextUnformatted("Graphics");
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(passName(node));
                    // Cadence carries both the author-declared policy and the
                    // live per-frame gating result: a scheduled pass shows its
                    // colored policy label, a pass skipped this frame (only
                    // possible for Once/OnResize/Invalidated) is dimmed and
                    // marked "(idle)". This folds in what used to be a separate
                    // "Run" column, which was fully redundant for Per-Frame.
                    ImGui::TableNextColumn();
                    const bool scheduled =
                        node < m_runtime->executeNodes.size() &&
                        m_runtime->executeNodes[node] != 0;
                    switch (passCadence(node))
                    {
                    case PassCadence::Once:
                        if (scheduled)
                            ImGui::TextColored(
                                ImVec4(0.55f, 0.75f, 1.0f, 1.0f), "Once");
                        else
                            ImGui::TextDisabled("Once (idle)");
                        break;
                    case PassCadence::OnResize:
                        if (scheduled)
                            ImGui::TextColored(
                                ImVec4(1.0f, 0.8f, 0.35f, 1.0f), "On Resize");
                        else
                            ImGui::TextDisabled("On Resize (idle)");
                        break;
                    case PassCadence::OnInvalidation:
                        if (scheduled)
                            ImGui::TextColored(
                                ImVec4(1.0f, 0.8f, 0.35f, 1.0f), "Invalidated");
                        else
                            ImGui::TextDisabled("Invalidated (idle)");
                        break;
                    case PassCadence::PerFrame:
                    default:
                        ImGui::TextDisabled("Per-Frame");
                        break;
                    }
                    ImGui::TableNextColumn();
                    if (batch.waitCount == 0)
                    {
                        ImGui::TextDisabled("-");
                    }
                    else
                    {
                        std::string waits;
                        for (uint32_t wait = 0; wait < batch.waitCount; ++wait)
                        {
                            const uint32_t sourceIndex =
                                plan.queueSubmissionWaits[
                                    batch.firstWait + wait].submissionIndex;
                            if (sourceIndex >= plan.queueSubmissions.size()) continue;
                            if (!waits.empty()) waits += "; ";
                            const QueueSubmissionBatch& source =
                                plan.queueSubmissions[sourceIndex];
                            waits += queueName(source.queue);
                            waits += ": ";
                            if (source.nodeCount == 0)
                            {
                                waits += "submission ";
                                waits += std::to_string(sourceIndex);
                            }
                            else
                            {
                                const GraphNodeId sourceNode =
                                    plan.queueSubmissionNodes[source.firstNode];
                                waits += passName(sourceNode);
                                if (source.nodeCount > 1)
                                {
                                    waits += " +";
                                    waits += std::to_string(source.nodeCount - 1);
                                    waits += " pass";
                                    if (source.nodeCount > 2) waits += "es";
                                }
                            }
                        }
                        ImGui::TextUnformatted(waits.c_str());
                    }
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(cpuParallel ? "Yes" : "No");
                    ImGui::TableNextColumn();
                    // Distinguish real overlap from mere eligibility:
                    //  - Overlap: this pass's level has another queue's batch, so
                    //    it truly executes concurrently on the GPU.
                    //  - Compute: on the async queue but nothing on another queue
                    //    shares its level (scheduled async, no overlap this level).
                    //  - Eligible: a compute-type pass sitting on the graphics
        //    queue (async disabled / unsupported) could overlap.
                    //  - dash: a graphics pass with no concurrent async work.
                    const bool isComputeType =
                        node < plan.nodes.size() &&
                        plan.nodes[node].type == GraphNodeType::Compute;
                    if (levelOverlaps)
                        ImGui::TextColored(
                            ImVec4(0.25f, 0.9f, 0.35f, 1.0f), "Overlap");
                    else if (batch.queue == QueueType::Compute)
                        ImGui::TextColored(
                            ImVec4(1.0f, 0.8f, 0.35f, 1.0f), "Compute");
                    else if (isComputeType)
                        ImGui::TextColored(
                            ImVec4(0.7f, 0.7f, 0.5f, 1.0f), "Eligible");
                    else
                        ImGui::TextDisabled("-");
                    ImGui::TableNextColumn();
                    ImGui::Text("%u", node);
                }
            }
            ImGui::EndTable();
        }
        ImGui::End();
    }
}
