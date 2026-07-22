#include "ic/common/ic_pch.h"
#include "ic/renderer/dx12_backend/dx12_backend.h"

#include "ic/core/frame_context.h"
#include "ic/renderer/frame_graph/frame_graph_executor.h"
#include "ic/core/app_base.h"
#include "ic/interface/window.h"
#include "ic/scene/scene_render_view.h"
#include "ic/renderer/pipeline_library.h"
#include "ic/renderer/gpu_driven_submission.h"
#include "ic/renderer/dx12_backend/dx12_pass_recorders.h"
#include "ic/renderer/dx12_backend/dx12_offscreen_pass_recorder.h"
#include "ic/renderer/dx12_backend/dx12_scene_pass_recorder.h"
#include "ic/renderer/dx12_backend/dx12_barrier_util.h"
#include "ic/renderer/renderer_specification.h"
#include "ic/renderer/path_tracing/path_trace_scene_builder.h"
#include "ic/renderer/renderer_common/renderer_util.h"
#include "ic/renderer/global_illumination/global_illumination_bindings.h"
#include "ic/renderer/global_illumination/global_illumination.h"

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <spdlog/spdlog.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_dx12.h>
#include <backends/imgui_impl_glfw.h>

#include <stdexcept>
#include <chrono>
#include <cmath>
#include <cstring>


namespace ic
{

    namespace
    {
        void throwIfFailed(HRESULT hr, const char* message)
        {
        if (FAILED(hr))
        {
            throw std::runtime_error(
                std::string(message) + " (HRESULT=" +
                std::to_string(static_cast<uint32_t>(hr)) + ")");
        }
        }

        uint32_t mipCountForSize(uint32_t size)
        {
            uint32_t levels = 1;
            while (size > 1)
            {
                size >>= 1u;
                ++levels;
            }
            return levels;
        }
    }


	void DX12Backend::init(
		const RendererSpecification& spec,
        const PipelineLibrary& pipelineLibrary,
		Window& window,
		uint32_t workerCount)
    {
        spdlog::info("[DX12Backend] Initializing...");
        m_giDebugView = static_cast<uint32_t>(
            spec.settings.globalIllumination.debugView);
        std::memcpy(&m_giDiagnosticIntensityBits,
            &spec.settings.globalIllumination.diagnosticIntensity,
            sizeof(m_giDiagnosticIntensityBits));
        std::memcpy(&m_giDebugExposureBits,
            &spec.settings.globalIllumination.debugExposure,
            sizeof(m_giDebugExposureBits));
        m_giFreezeAfterFrames =
            spec.settings.globalIllumination.freezeAfterFrames;
        m_giMaxSurfelUpdates =
            spec.settings.globalIllumination.limits.maxSurfelUpdates;
        m_giMaxProbeUpdates =
            spec.settings.globalIllumination.limits.maxProbeUpdates;
        m_giRayBudget = spec.settings.globalIllumination.limits.rayBudget;

#ifdef IC_DEBUG
        constexpr bool validationAvailable = true;
#else
        constexpr bool validationAvailable = false;
#endif
        m_factory.init(spec.enableValidation && validationAvailable);
        m_adapter.init(m_factory);
        m_device.init(m_adapter, m_factory.validationEnabled());
        m_resourceAllocator.init(m_device);

        const uint32_t framesInFlight =
            spec.framesInFlight == 0 ? 1 : spec.framesInFlight;

        const uint32_t workerSlots =
            workerCount == 0 ? 1 : workerCount;
        m_workerSlots = workerSlots;

        m_swapchain.setVsyncEnabled(spec.settings.vsync);

        // A flip-model swapchain needs at least one more back buffer than the
        // number of frames the CPU may have in flight, otherwise every present
        // stalls waiting for a buffer to free. With VSync, this collapses to
        // half the refresh rate. framesInFlight + 1 gives full-rate pacing.
        m_swapchain.init(
            m_factory,
            m_device,
            window,
            framesInFlight + 1u);
        // Seed the authoritative framebuffer size so the first reconcile does
        // not spuriously recreate a just-created swapchain.
        m_lastFramebufferWidth = m_swapchain.width();
        m_lastFramebufferHeight = m_swapchain.height();

        m_frameExecutor.init(m_device, m_swapchain, framesInFlight);

        D3D12_QUERY_HEAP_DESC timestampHeapDesc{};
        timestampHeapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        timestampHeapDesc.Count = framesInFlight * 2u;
        throwIfFailed(
            m_device.device()->CreateQueryHeap(
                &timestampHeapDesc,
                IID_PPV_ARGS(&m_gpuCullTimestampHeap)),
            "Failed to create DX12 GPU-cull timestamp heap.");
        m_gpuCullTimestampReadback = m_resourceAllocator.createBuffer({
            .size = sizeof(uint64_t) * framesInFlight * 2u,
            .usage = BufferUsageFlags::TransferDst,
            .memoryUsage = ResourceMemoryUsage::GpuToCpu,
            .mappedAtCreation = true,
            .debugName = "GPU cull timestamp readback"
        });
        throwIfFailed(
            m_device.graphicsQueue()->GetTimestampFrequency(
                &m_gpuCullTimestampFrequency),
            "Failed to query DX12 compute timestamp frequency.");
        m_gpuCullCpuRecordMilliseconds.assign(framesInFlight, 0.0);
        m_gpuCullTimestampValid.assign(framesInFlight, uint8_t{0});

        // Budget covers every pass of the largest demo graph with headroom;
        // passes beyond it simply record untimed rather than failing.
        m_gpuProfiler.init(
            m_device, m_resourceAllocator, framesInFlight, 128u);

        buildBackendDiagnostics();

        m_commandSystem.init(
            m_device.device(),
            framesInFlight,
            workerSlots);

        m_descriptorSystem.init(m_device);
        m_graphResourceRegistry.init(
            m_device,
            m_resourceAllocator,
            m_descriptorSystem,
            framesInFlight);
        // The graph-owned GPU-driven indirect-argument buffer is sized in native
        // DX12 command units (draw args + embedded root constants).
        m_graphResourceRegistry.setNativeIndirectCommandStride(
            sizeof(DX12GpuIndexedIndirectCommand));
        m_gpuScene.init(
            m_device,
            m_resourceAllocator,
            m_descriptorSystem,
            framesInFlight);
        D3D12_INDIRECT_ARGUMENT_DESC dispatchArgument{};
        dispatchArgument.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
        D3D12_COMMAND_SIGNATURE_DESC dispatchSignature{};
        dispatchSignature.ByteStride = sizeof(D3D12_DISPATCH_ARGUMENTS);
        dispatchSignature.NumArgumentDescs = 1;
        dispatchSignature.pArgumentDescs = &dispatchArgument;
        throwIfFailed(
            m_device.device()->CreateCommandSignature(
                &dispatchSignature,
                nullptr,
                IID_PPV_ARGS(&m_dispatchIndirectSignature)),
            "Failed to create DX12 dispatch-indirect command signature.");
        m_uploadScheduler.init(
            m_device, m_resourceAllocator, framesInFlight);
        m_retirementQueue.init(
            m_resourceAllocator, m_descriptorSystem, framesInFlight);
        m_accelerationStructures.init(
            m_device, m_resourceAllocator, m_retirementQueue, framesInFlight);
        m_pipelineManager.init(m_device);
        m_pipelineLibrary = &pipelineLibrary;

        if (spec.useDebugGui)
        {
            initImGui(window);
        }

        spdlog::info("[DX12Backend] Initialized");
    }

    void DX12Backend::shutdown()
    {
        m_frameExecutor.waitForGpu();
        m_uploadScheduler.shutdown();
        m_accelerationStructures.shutdown();
        m_retirementQueue.drain();

        shutdownImGui();
        destroySceneResources();
        destroyEnvironmentResources();
        destroyClusteredForwardResources();
        destroyPathTraceResources();
        m_gpuProfiler.shutdown();
        m_dispatchIndirectSignature.Reset();
        m_gpuCullTimestampHeap.Reset();
        m_resourceAllocator.destroyBuffer(m_gpuCullTimestampReadback);
        m_graphResourceRegistry.shutdown();
        m_frameExecutor.shutdown();
        m_pipelineManager.shutdown();
        m_descriptorSystem.shutdown();
        m_resourceAllocator.shutdown();
        m_commandSystem.shutdown();
        m_swapchain.shutdown();
        m_device.shutdown();
        m_adapter.shutdown();
        m_factory.shutdown();

        spdlog::info("[DX12Backend] Shutdown");
    }

    void DX12Backend::drainForSchedulingTransition()
    {
        // Rare control-plane transition only. Waiting all frame slots here is
        // required because the replacement graph may move resource accesses
        // and cross-frame edges to a different queue.
        m_frameExecutor.waitForGpu();
    }

    RenderSurfaceState DX12Backend::reconcileRenderSurface()
    {
        RenderSurfaceState surface{};

        uint32_t framebufferWidth = 0;
        uint32_t framebufferHeight = 0;
        m_swapchain.windowFramebufferSize(framebufferWidth, framebufferHeight);

        // Minimized / zero drawable area: leave the swapchain untouched and
        // report the frame as non-renderable so the renderer skips it entirely.
        if (framebufferWidth == 0 || framebufferHeight == 0)
        {
            return surface;
        }

        // Recreate when the framebuffer changed or the swapchain is no longer
        // valid (e.g. a prior out-of-date present). recreateSwapchain bumps the
        // generation, so the renderer rebuilds the graph to the new extent and
        // re-materializes every resource before this frame records anything.
        if (framebufferWidth != m_lastFramebufferWidth ||
            framebufferHeight != m_lastFramebufferHeight ||
            !m_swapchain.validForRendering())
        {
            recreateSwapchain();
            m_lastFramebufferWidth = framebufferWidth;
            m_lastFramebufferHeight = framebufferHeight;
        }

        if (!m_swapchain.validForRendering())
        {
            return surface;
        }

        surface.width = m_swapchain.width();
        surface.height = m_swapchain.height();
        surface.generation = m_swapchainGeneration;
        surface.renderable = true;
        return surface;
    }

    bool DX12Backend::execute(
        const CompiledGraphPlan& plan,
        const GraphExecutionContext& execution,
        const FrameContext& ctx,
        const SceneRenderView& scene)
    {
        m_performanceCounters = {};
        const auto backendStart = std::chrono::steady_clock::now();
        if (!m_frameExecutor.ready())
        {
            return false;
        }

        const uint32_t frameSlot =
            static_cast<uint32_t>(
                ctx.frameIndex % m_frameExecutor.framesInFlight());

        const auto waitStart = std::chrono::steady_clock::now();
        m_frameExecutor.waitForFrame(frameSlot);
        m_performanceCounters.frameSlotWaitMs =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - waitStart).count();
        // Safe only after the slot's fence wait above: this reads back the
        // timestamps recorded into this slot framesInFlight frames ago.
        const auto profilerStart = std::chrono::steady_clock::now();
        m_gpuProfiler.beginFrame(frameSlot);
        m_performanceCounters.profilerReadbackMs =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - profilerStart).count();
        if (m_gpuCullTimestampValid[frameSlot] != 0u &&
            m_gpuCullTimestampReadback.mapped &&
            m_gpuCullTimestampFrequency != 0)
        {
            const auto* timestamps = static_cast<const uint64_t*>(
                m_gpuCullTimestampReadback.mapped) + frameSlot * 2u;
            m_gpuCullPerformance.gpuMilliseconds =
                static_cast<double>(timestamps[1] - timestamps[0]) *
                1000.0 / static_cast<double>(m_gpuCullTimestampFrequency);
            m_gpuCullPerformance.cpuRecordMilliseconds =
                m_gpuCullCpuRecordMilliseconds[frameSlot];
        }
        m_retirementQueue.beginFrame(frameSlot);

        // The swapchain was already reconciled to the window this frame in
        // reconcileRenderSurface(), and the graph was (re)built to match the
        // extent it returned. This is a pure guard: no size change or recreate
        // happens mid-frame, so the backbuffer and graph attachments cannot
        // diverge in extent.
        if (!m_swapchain.validForRendering())
        {
            return false;
        }

        m_uploadScheduler.beginFrame(frameSlot);
        m_commandSystem.beginFrame(frameSlot);

        ID3D12Resource* swapchainImage =
            m_swapchain.currentBackBuffer();

        // The slot's prior GPU work completed in waitForFrame above, so any
        // resources retired the last time this slot was used are now safe to
        // free without adding a GPU wait.
        m_graphResourceRegistry.recycleFrameSlot(frameSlot);

        DX12GraphResourceImports graphImports{};
        graphImports.swapchainResource = swapchainImage;
        graphImports.swapchainRtv = m_swapchain.currentRtv();
        m_graphResourceRegistry.materialize(
            plan,
            frameSlot,
            m_swapchain.width(),
            m_swapchain.height(),
            graphImports);
        const GraphResourceId giReadbackId = findResourceBySemantic(
            plan, GraphResourceSemantic::GiDiagnosticsReadback);
        m_giDiagnosticsReadbackActive =
            giReadbackId != InvalidGraphResourceId;
        if (giReadbackId != InvalidGraphResourceId)
        {
            // Delay the one-shot report until the persistent cache has had a
            // representative warm-up window instead of logging its first,
            // deliberately sparse population frame.
            const bool freezeReportReady = m_giFreezeAfterFrames == 0u ||
                ctx.frameIndex >= m_giCacheInitializationFrame +
                    m_giFreezeAfterFrames + 10u;
            if (freezeReportReady && m_giDiagnosticFrames >=
                m_frameExecutor.framesInFlight() + 120u)
            {
                DX12GraphResourceEntry* readback =
                    m_graphResourceRegistry.entry(giReadbackId);
                if (readback && readback->buffer.mapped)
                {
                    std::memcpy(m_giDiagnosticWords.data(),
                        readback->buffer.mapped, sizeof(GpuGiDiagnostics));
                    GpuGiDiagnostics diagnostics{};
                    std::memcpy(&diagnostics, m_giDiagnosticWords.data(),
                        sizeof(diagnostics));
                    if (!m_loggedGiDiagnostics &&
                        (diagnostics.tracedRays != 0u ||
                         diagnostics.validPixelCount != 0u ||
                         diagnostics.surfelOccupancy != 0u ||
                         diagnostics.allocationFailures != 0u ||
                         diagnostics.rejectedGathers != 0u))
                    {
                        float maxIrradiance = 0.0f;
                        std::memcpy(&maxIrradiance,
                            &diagnostics.maxIrradianceBits, sizeof(float));
                        const float averageIrradiance = diagnostics.validPixelCount
                            ? static_cast<float>(diagnostics.irradianceLuminanceFixed) /
                                (256.0f * diagnostics.validPixelCount) : 0.0f;
                        const uint32_t evaluatedPixels =
                            diagnostics.validPixelCount + diagnostics.rejectedGathers;
                        const float validCoverage = evaluatedPixels
                            ? 100.0f * diagnostics.validPixelCount / evaluatedPixels : 0.0f;
                        const float averageConfidence = diagnostics.validPixelCount
                            ? static_cast<float>(diagnostics.coverageFixed) /
                                (4096.0f * diagnostics.validPixelCount) : 0.0f;
                        const float averageReferenceError = diagnostics.referenceCount
                            ? static_cast<float>(diagnostics.referenceErrorFixed) /
                                (4096.0f * diagnostics.referenceCount) : 0.0f;
                        const float channelDenominator = diagnostics.validPixelCount
                            ? 256.0f * diagnostics.validPixelCount : 1.0f;
                        const glm::vec3 averageIrradianceRgb{
                            diagnostics.irradianceRedFixed / channelDenominator,
                            diagnostics.irradianceGreenFixed / channelDenominator,
                            diagnostics.irradianceBlueFixed / channelDenominator };
                        spdlog::info(
                            "[DX12 GI] surfels={} probes={} probeUpdates={} feedbackSamples={} probeFallbackPixels={} allocationFailures={} rejectedGathers={} rays={} hits={} misses={} alphaRejects={} stale={} invalid={} selfRejects={} updates={} avgIrradiance={} avgIrradianceRgb=({},{},{}) maxIrradiance={} validCoveragePct={} avgConfidence={} unitDiffuseInjectedLuma={} temporalRejected={} referenceSamples={} avgShReferenceError={}",
                            diagnostics.surfelOccupancy,
                            diagnostics.probeOccupancy,
                            diagnostics.probeUpdateCount,
                            diagnostics.feedbackSamples,
                            diagnostics.probeFallbackPixels,
                            diagnostics.allocationFailures,
                            diagnostics.rejectedGathers,
                            diagnostics.tracedRays,
                            diagnostics.hits,
                            diagnostics.misses,
                            diagnostics.alphaRejections,
                            diagnostics.staleSurfels,
                            diagnostics.invalidMappings,
                            diagnostics.selfHitRejections,
                            diagnostics.surfelUpdateCount,
                            averageIrradiance, averageIrradianceRgb.r,
                            averageIrradianceRgb.g, averageIrradianceRgb.b,
                            maxIrradiance, validCoverage,
                            averageConfidence, averageIrradiance / 3.14159265359f,
                            diagnostics.temporalRejectedPixels,
                            diagnostics.referenceCount, averageReferenceError);
                        m_loggedGiDiagnostics = true;
                    }
                }
            }
            ++m_giDiagnosticFrames;
        }
        else
        {
            m_giDiagnosticFrames = 0;
            m_loggedGiDiagnostics = false;
            m_giDiagnosticWords.fill(0u);
        }
        const GraphResourceId statsReadbackId = findResourceBySemantic(
            plan, GraphResourceSemantic::GpuDrivenCullStatsReadback);
        if (statsReadbackId != InvalidGraphResourceId &&
            m_gpuCullDiagnosticFrames >= m_frameExecutor.framesInFlight())
        {
            DX12GraphResourceEntry* readback =
                m_graphResourceRegistry.entry(statsReadbackId);
            if (readback && readback->buffer.mapped)
            {
                std::memcpy(
                    &m_gpuCullStats,
                    readback->buffer.mapped,
                    sizeof(m_gpuCullStats));
                if (m_gpuCullDiagnosticFrames % 600u == 0u)
                {
                    spdlog::info(
                        "[DX12Backend] GPU cull: input={} visible={} frustum={} occlusion={} conservative={} falseOccluded={} falseVisible={} overflow={} history={} gpu={:.3f}ms cpuRecord={:.3f}ms",
                        m_gpuCullStats.inputCount,
                        m_gpuCullStats.visible,
                        m_gpuCullStats.frustumCulled,
                        m_gpuCullStats.occlusionCulled,
                        m_gpuCullStats.conservativeRetained,
                        m_gpuCullStats.falseOccluded,
                        m_gpuCullStats.falseVisible,
                        m_gpuCullStats.overflow,
                        m_gpuCullStats.historyValid,
                        m_gpuCullPerformance.gpuMilliseconds,
                        m_gpuCullPerformance.cpuRecordMilliseconds);
                }
            }
        }

        std::vector<ID3D12CommandList*>& commandLists = m_frameCommandLists;
        commandLists.clear();
        const auto graphStart = std::chrono::steady_clock::now();
        executeGraph(
            plan,
            execution,
            ctx,
            scene,
            swapchainImage,
            commandLists);
        m_performanceCounters.graphRecordMs =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - graphStart).count();
        const auto uiStart = std::chrono::steady_clock::now();
        recordImGui(
            ctx,
            swapchainImage,
            commandLists);
        m_performanceCounters.uiRecordMs =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - uiStart).count();
        const auto validationStart = std::chrono::steady_clock::now();
        m_device.logValidationMessages();
        m_performanceCounters.validationMs =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - validationStart).count();

        const DX12UploadDependency uploadDependency =
            m_uploadScheduler.flush();
        const auto submitStart = std::chrono::steady_clock::now();
        const bool presented = m_frameExecutor.submitAndPresent(
            plan, commandLists, frameSlot, execution, uploadDependency);
        m_performanceCounters.submitPresentMs =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - submitStart).count();
        if (m_gpuCullDebugMode != GpuCullDebugMode::Off)
        {
            ++m_gpuCullDiagnosticFrames;
        }

        if (!presented)
        {
            recreateSwapchain();
        }
        m_performanceCounters.backendCpuMs =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - backendStart).count();
        return presented;
    }


    void DX12Backend::executeGraph(
        const CompiledGraphPlan& plan,
        const GraphExecutionContext& execution,
        const FrameContext& ctx,
        const SceneRenderView& scene,
        ID3D12Resource* swapchainImage,
        std::vector<ID3D12CommandList*>& commandLists)
    {
        // Resolve lazy pipeline/resource state before worker threads begin.
        // Recording jobs may only read this shared renderer state.
        m_resolvedDiffuseGi = findResourceBySemantic(
            plan, GraphResourceSemantic::GiResolvedIrradiance);
        ensureDepthTarget();
        ensureClusteredForwardResources();
        for (const ExecutionNode& node : plan.nodes)
        {
            if (node.type == GraphNodeType::Graphics)
            {
                (void)pipelineForNode(plan, node);
                const GraphicsPassData* pass =
                    node.payloadIndex < plan.payloads.size()
                        ? std::get_if<GraphicsPassData>(
                            &plan.payloads[node.payloadIndex])
                        : nullptr;
                if (pass && pass->drawList == DrawListKind::SceneGeometry)
                {
                    (void)prepareSceneResources(ctx, scene);
                }
            }
            else if (node.type == GraphNodeType::Compute)
            {
                const bool nativeAsPass = node.payloadIndex < plan.payloads.size() &&
                    std::holds_alternative<AccelerationStructureBuildPassData>(
                        plan.payloads[node.payloadIndex]);
                if (nativeAsPass)
                    (void)prepareSceneResources(ctx, scene);
                else
                    (void)computePipelineForNode(plan, node);
            }
        }

        // Upload this frame's cull inputs into the graph-owned per-frame-slot
        // buffers before any worker records the cull dispatch. Serial, after
        // prepareSceneResources has populated the prepared scene above.
        uploadGpuDrivenInputs(plan, ctx);
        if (m_resolvedDiffuseGi != InvalidGraphResourceId)
            (void)prepareGiRayQueryResources(ctx);

        auto recordNode =
            [&](GraphNodeId nodeId, uint32_t workerIndex)
            {
                auto lease =
                    m_commandSystem.acquireFrameCommandList(
                        static_cast<uint32_t>(ctx.frameIndex % m_frameExecutor.framesInFlight()),
                        workerIndex,
                        plan.nodes[nodeId].queue);

                ID3D12GraphicsCommandList4* cmd =
                    lease.commandList();

                const ExecutionNode& node = plan.nodes[nodeId];

                const uint32_t frameSlot = static_cast<uint32_t>(
                    ctx.frameIndex % m_frameExecutor.framesInFlight());

                // Time the pass including its barriers: a cross-queue acquire
                // that stalls the queue is part of what the pass costs, and
                // hiding it would flatter the async path.
                const uint32_t profileSlot = m_gpuProfiler.beginPass(
                    cmd, frameSlot, node.nodeId, node.queue);

                applyBarriers(
                    cmd,
                    plan,
                    node,
                    swapchainImage);

                if (execution.shouldExecute(node.nodeId))
                {
                    dispatchNode(
                        plan,
                        node,
                        ctx,
                        scene,
                        cmd,
                        swapchainImage);
                }

                if (node.nodeId < plan.nodeSchedules.size())
                {
                    const NodeSchedule& schedule =
                        plan.nodeSchedules[node.nodeId];
                    for (uint32_t i = 0;
                         i < schedule.outgoingBarrierCount; ++i)
                    {
                        const uint32_t barrierIndex =
                            plan.outgoingBarrierIndices[
                                schedule.firstOutgoingBarrier + i];
                        const ResourceBarrier& barrier =
                            plan.barriers[barrierIndex];
                        if (barrier.fromNode != barrier.toNode &&
                            plan.nodes[barrier.fromNode].queue !=
                                plan.nodes[barrier.toNode].queue)
                        {
                            recordBarrier(
                                cmd, barrier, plan.resources, swapchainImage,
                                true, false, node.queue);
                        }
                    }
                }

                m_gpuProfiler.endPass(
                    cmd, frameSlot, node.queue, profileSlot);

                const HRESULT closeResult = cmd->Close();
                if (FAILED(closeResult))
                {
                    m_device.logValidationMessages();
                    throwIfFailed(
                        closeResult,
                        "Failed to close DX12 frame command list.");
                }

                return static_cast<ID3D12CommandList*>(cmd);
            };

        recordFrameGraph(
            plan,
            ctx.services ? ctx.services->jobSystem : nullptr,
            m_workerSlots,
            recordNode,
            commandLists);
    }

    void DX12Backend::applyBarriers(
        ID3D12GraphicsCommandList4* cmd,
        const CompiledGraphPlan& plan,
        const ExecutionNode& node,
        ID3D12Resource* swapchainImage)
    {
        if (node.nodeId >= plan.nodeSchedules.size())
        {
            for (const ResourceBarrier& barrier : plan.barriers)
            {
                if (barrier.toNode != node.nodeId)
                {
                    continue;
                }

                recordBarrier(
                    cmd,
                    barrier,
                    plan.resources,
                    swapchainImage);
            }

            return;
        }

        const NodeSchedule& schedule =
            plan.nodeSchedules[node.nodeId];

        for (uint32_t i = 0; i < schedule.incomingBarrierCount; ++i)
        {
            const uint32_t barrierIndex =
                plan.incomingBarrierIndices[
                    schedule.firstIncomingBarrier + i];
            if (barrierIndex >= plan.barriers.size())
            {
                continue;
            }

            const ResourceBarrier& barrier = plan.barriers[barrierIndex];
            recordBarrier(
                cmd,
                barrier,
                plan.resources,
                swapchainImage,
                false,
                barrier.fromNode != barrier.toNode &&
                    plan.nodes[barrier.fromNode].queue !=
                        plan.nodes[barrier.toNode].queue,
                node.queue);
        }
    }

    void DX12Backend::recordBarrier(
        ID3D12GraphicsCommandList4* cmd,
        const ResourceBarrier& barrier,
        std::span<const GraphResource> resources,
        ID3D12Resource* swapchainImage,
        bool crossQueueRelease,
        bool crossQueueAcquire,
        QueueType commandQueue)
    {
        const GraphResource& resource =
            resources[barrier.resource];

        ID3D12Resource* dxResource = nullptr;

        if (resource.ownership == ResourceOwnership::Imported)
        {
            switch (resource.imported)
            {
            case ImportedResource::Swapchain:
                dxResource = swapchainImage;
                break;

            case ImportedResource::None:
                break;
            }
        }
        DX12GraphResourceEntry* graphEntry = nullptr;
        if (resource.ownership != ResourceOwnership::Imported)
        {
            // A previous-version barrier transitions the previous frame's
            // physical instance of a history resource, not the current one.
            graphEntry = barrier.previousVersion
                ? m_graphResourceRegistry.previousEntry(barrier.resource)
                : m_graphResourceRegistry.entry(barrier.resource);
            if (graphEntry)
            {
                dxResource = graphEntry->type == GraphResourceType::Texture
                    ? graphEntry->texture.resource.Get()
                    : graphEntry->buffer.resource.Get();
            }
        }

        if (!dxResource)
        {
            spdlog::error(
                "[DX12Backend] Missing graph resource handle for barrier resource {}",
                barrier.resource);
            return;
        }

        // Resolve the resource state for a usage on the queue this barrier is
        // being recorded on. The compute queue cannot use the graphics-only
        // PIXEL_SHADER_RESOURCE bit, so a sampled read there resolves to the
        // non-pixel state only.
        //
        // Graph allocations persist across frames, so the compiled first-use
        // state is only correct immediately after materialization. Carry the
        // native state in the registry thereafter. Resource-conflicting nodes
        // and read-only usage changes have compiler dependencies, so updates
        // for one resource occur in sequential recording levels.
        auto stateForUsage =
            [&](ResourceUsage usage) -> D3D12_RESOURCE_STATES
            {
                if (commandQueue == QueueType::Compute &&
                    usage == ResourceUsage::SampledTexture)
                {
                    return D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                }
                return usageToState(usage);
            };

        // Buffers are never explicitly transitioned. D3D12 buffers implicitly
        // promote from COMMON to whatever state an access needs and decay back
        // to COMMON after every ExecuteCommandLists, on any queue. This is the
        // single, correct policy for graph-owned buffers:
        //   - No explicit transition barrier can go stale against the driver's
        //     implicit decay (the graph does not model decay), and none asserts
        //     a cross-queue before-state the other queue cannot validate. The
        //     exact source of the DXGI_ERROR_ACCESS_DENIED device removal when a
        //     buffer produced on one queue is consumed on another.
        //   - A UAV barrier still orders write -> read/write on the same queue.
        //     Cross-queue ordering + memory visibility come from the executor's
        //     queue fence, so no barrier is emitted for the cross-queue edge.
        // The tracked state is pinned to COMMON so any later same-queue barrier
        // for this buffer starts from the state the driver actually decays to.
        // (Textures still transition explicitly below because they have real layouts.)
        if (resource.type == GraphResourceType::Buffer)
        {
            const bool writeHazard =
                !crossQueueRelease && !crossQueueAcquire &&
                (barrier.fromAccess != AccessType::Read ||
                 barrier.toAccess != AccessType::Read) &&
                hasFlag(
                    resource.bufferDesc.usage,
                    BufferUsageFlags::Storage) &&
                resource.bufferDesc.memoryUsage ==
                    ResourceMemoryUsage::GpuOnly;
            if (writeHazard)
            {
                D3D12_RESOURCE_BARRIER uav{};
                uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                uav.UAV.pResource = dxResource;
                cmd->ResourceBarrier(1, &uav);
            }
            if (graphEntry) graphEntry->state = D3D12_RESOURCE_STATE_COMMON;
            return;
        }

        const bool tracked = graphEntry != nullptr;
        const D3D12_RESOURCE_STATES before = tracked
            ? graphEntry->state
            : (crossQueueAcquire || barrier.firstUse)
            ? D3D12_RESOURCE_STATE_COMMON
            : stateForUsage(barrier.oldUsage);

        const D3D12_RESOURCE_STATES after = crossQueueRelease
            ? D3D12_RESOURCE_STATE_COMMON
            : stateForUsage(barrier.newUsage);

        if (before == after)
        {
            if (before == D3D12_RESOURCE_STATE_UNORDERED_ACCESS &&
                (barrier.fromAccess != AccessType::Read ||
                 barrier.toAccess != AccessType::Read))
            {
                D3D12_RESOURCE_BARRIER uav{};
                uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                uav.UAV.pResource = dxResource;
                cmd->ResourceBarrier(1, &uav);
            }
            if (tracked) graphEntry->state = after;
            return;
        }

        transitionResource(
            cmd,
            dxResource,
            before,
            after);
        if (tracked) graphEntry->state = after;

    }

    DX12PassContext DX12Backend::makePassContext(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node,
        const FrameContext& ctx,
        const SceneRenderView& scene,
        ID3D12GraphicsCommandList4* cmd,
        ID3D12Resource* swapchainImage)
    {
        DX12PassContext passCtx{};
        passCtx.cmd = cmd;
        passCtx.plan = &plan;
        passCtx.node = &node;
        passCtx.frame = &ctx;
        passCtx.scene = &scene;
        passCtx.resources = &m_graphResourceRegistry;
        passCtx.descriptors = &m_descriptorSystem;
        passCtx.pipelines = &m_pipelineManager;
        passCtx.gpuScene = &m_gpuScene;
        passCtx.swapchainImage = swapchainImage;
        passCtx.surfaceWidth = m_swapchain.width();
        passCtx.surfaceHeight = m_swapchain.height();
        passCtx.fallbackRtv = m_swapchain.currentRtv();
        passCtx.fallbackDsv = m_depthDsv.cpuStart;
        return passCtx;
    }

    void DX12Backend::dispatchNode(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node,
        const FrameContext& ctx,
        const SceneRenderView& scene,
        ID3D12GraphicsCommandList4* cmd,
        ID3D12Resource* swapchainImage)
    {
        if (node.payloadIndex >= plan.payloads.size())
        {
            return;
        }

        // Exhaustive payload dispatch: the visit binds the node's variant to the
        // matching recordPassPayload overload. Overload resolution replaces the
        // former node.type switch and the compute-node get_if cascade in one
        // place, and a new PassPayload alternative without an overload is a
        // compile error instead of a silently skipped pass.
        std::visit(
            [&](const auto& payload)
            {
                recordPassPayload(
                    payload, plan, node, ctx, scene, cmd, swapchainImage);
            },
            plan.payloads[node.payloadIndex]);
    }

    void DX12Backend::recordPassPayload(
        const GraphicsPassData&,
        const CompiledGraphPlan& plan, const ExecutionNode& node,
        const FrameContext& ctx, const SceneRenderView& scene,
        ID3D12GraphicsCommandList4* cmd, ID3D12Resource* swapchainImage)
    {
        executeGraphicsNode(plan, node, ctx, scene, cmd, swapchainImage);
    }

    void DX12Backend::recordPassPayload(
        const ComputePassData&,
        const CompiledGraphPlan& plan, const ExecutionNode& node,
        const FrameContext& ctx, const SceneRenderView& scene,
        ID3D12GraphicsCommandList4* cmd, ID3D12Resource*)
    {
        executeComputeNode(plan, node, ctx, scene, cmd);
    }

    void DX12Backend::recordPassPayload(
        const PathTracePassData&,
        const CompiledGraphPlan& plan, const ExecutionNode& node,
        const FrameContext& ctx, const SceneRenderView& scene,
        ID3D12GraphicsCommandList4* cmd, ID3D12Resource*)
    {
        executePathTraceNode(plan, node, ctx, scene, cmd);
    }

    void DX12Backend::recordPassPayload(
        const TonemapPassData&,
        const CompiledGraphPlan& plan, const ExecutionNode& node,
        const FrameContext& ctx, const SceneRenderView&,
        ID3D12GraphicsCommandList4* cmd, ID3D12Resource*)
    {
        executeTonemapNode(plan, node, ctx, cmd);
    }

    void DX12Backend::recordPassPayload(
        const EnvironmentConvertPassData&,
        const CompiledGraphPlan& plan, const ExecutionNode& node,
        const FrameContext& ctx, const SceneRenderView& scene,
        ID3D12GraphicsCommandList4* cmd, ID3D12Resource*)
    {
        executeEnvironmentConvertNode(plan, node, ctx, scene, cmd);
    }

    void DX12Backend::recordPassPayload(
        const TransferPassData&,
        const CompiledGraphPlan& plan, const ExecutionNode& node,
        const FrameContext& ctx, const SceneRenderView&,
        ID3D12GraphicsCommandList4* cmd, ID3D12Resource* swapchainImage)
    {
        executeTransferNode(plan, node, ctx, cmd, swapchainImage);
    }

    void DX12Backend::recordPassPayload(
        const AccelerationStructureBuildPassData&,
        const CompiledGraphPlan&, const ExecutionNode&,
        const FrameContext& ctx, const SceneRenderView&,
        ID3D12GraphicsCommandList4* cmd, ID3D12Resource*)
    {
        if (!m_rayTracingSceneService) return;
        m_accelerationStructures.recordBuild(
            cmd, *m_rayTracingSceneService, m_uploadedModels,
            static_cast<uint32_t>(ctx.frameIndex %
                m_frameExecutor.framesInFlight()));
    }

    void DX12Backend::executeGraphicsNode(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node,
        const FrameContext& ctx,
        const SceneRenderView& scene,
        ID3D12GraphicsCommandList4* cmd,
        [[maybe_unused]] ID3D12Resource* swapchainImage)
    {
        DX12GraphicsPipeline* pipeline =
            m_pipelineManager.graphicsPipeline(
                pipelineForNode(plan, node));
        if (!pipeline)
        {
            return;
        }

        const GraphicsPassData* pass =
            node.payloadIndex < plan.payloads.size()
                ? std::get_if<GraphicsPassData>(
                    &plan.payloads[node.payloadIndex])
                : nullptr;
        const bool usesClusterData =
            nodeUsesResourceSemantic(
                plan, node, GraphResourceSemantic::ClusterBounds) ||
            nodeUsesResourceSemantic(
                plan, node, GraphResourceSemantic::ClusterLightGrid) ||
            nodeUsesResourceSemantic(
                plan, node, GraphResourceSemantic::ClusterLightIndices) ||
            nodeUsesResourceSemantic(
                plan, node, GraphResourceSemantic::ClusterLightCounter);

        const bool hasColorTarget =
            pipeline->desc.colorAttachmentCount > 0;
        const GraphResourceId colorResource =
            findNodeResource(plan, node, ResourceUsage::ColorAttachment);
        const GraphResourceId depthResource =
            findNodeResource(plan, node, ResourceUsage::DepthAttachment);
        const DX12GraphResourceEntry* colorEntry =
            colorResource != InvalidGraphResourceId
                ? m_graphResourceRegistry.entry(colorResource)
                : nullptr;
        const DX12GraphResourceEntry* depthEntry =
            depthResource != InvalidGraphResourceId
                ? m_graphResourceRegistry.entry(depthResource)
                : nullptr;
        constexpr bool useGraphAttachments = true;

        const D3D12_CPU_DESCRIPTOR_HANDLE rtv =
            useGraphAttachments &&
                colorEntry && colorEntry->ownership != ResourceOwnership::Imported &&
                colorEntry->rtv.valid()
                ? colorEntry->rtv.cpuStart
                : m_swapchain.currentRtv();

        D3D12_VIEWPORT viewport{};
        viewport.TopLeftX = 0.0f;
        viewport.TopLeftY = 0.0f;
        viewport.Width = static_cast<float>(m_swapchain.width());
        viewport.Height = static_cast<float>(m_swapchain.height());
        viewport.MinDepth = 0.0f;
        viewport.MaxDepth = 1.0f;

        D3D12_RECT scissor{};
        scissor.left = 0;
        scissor.top = 0;
        scissor.right = static_cast<LONG>(m_swapchain.width());
        scissor.bottom = static_cast<LONG>(m_swapchain.height());

        cmd->RSSetViewports(1, &viewport);
        cmd->RSSetScissorRects(1, &scissor);

        if (!useGraphAttachments || !depthEntry)
        {
            ensureDepthTarget();
            if (m_depthState != D3D12_RESOURCE_STATE_DEPTH_WRITE)
            {
                transitionResource(
                    cmd,
                    m_depthTexture.resource.Get(),
                    m_depthState,
                    D3D12_RESOURCE_STATE_DEPTH_WRITE);
                m_depthState = D3D12_RESOURCE_STATE_DEPTH_WRITE;
            }
        }

        const D3D12_CPU_DESCRIPTOR_HANDLE dsv =
            useGraphAttachments &&
                depthEntry && depthEntry->ownership != ResourceOwnership::Imported &&
                depthEntry->dsv.valid()
                ? depthEntry->dsv.cpuStart
                : m_depthDsv.cpuStart;

        cmd->OMSetRenderTargets(
            hasColorTarget ? 1u : 0u,
            hasColorTarget ? &rtv : nullptr,
            FALSE,
            &dsv);

        constexpr FLOAT clearColor[4] = { 0.02f, 0.02f, 0.025f, 1.0f };
        if (hasColorTarget &&
            (!pass || pass->colorLoadOp == AttachmentLoadOp::Clear))
        {
            cmd->ClearRenderTargetView(rtv, clearColor, 0, nullptr);
        }
        if (!hasColorTarget ||
            (pass && pass->depthLoadOp == AttachmentLoadOp::Clear))
        {
            cmd->ClearDepthStencilView(
                dsv,
                D3D12_CLEAR_FLAG_DEPTH,
                0.0f,
                0,
                0,
                nullptr);
        }

        if (pass && pass->drawList == DrawListKind::Skybox)
        {
            drawSkybox(*pipeline, ctx, scene, cmd);
            return;
        }

        if (!pass || pass->drawList != DrawListKind::SceneGeometry)
        {
            return;
        }

        if (!m_environmentResources.converted)
        {
            if (DX12ComputePipeline* convertPipeline =
                    environmentConvertPipeline())
            {
                (void)convertEnvironmentIfReady(
                    *convertPipeline,
                    ctx,
                    scene,
                    cmd);
            }
        }

        if (!prepareSceneResources(ctx, scene) ||
            m_gpuScene.draws().empty())
        {
            return;
        }

        const uint32_t frameSlot =
            static_cast<uint32_t>(ctx.frameIndex % m_gpuScene.frameSlotCount());
        DX12GpuSceneFrameResources& frameResources =
            m_gpuScene.frameResources(frameSlot);

        // Resolve the GPU-driven indirect stream (graph-registry owned). The
        // indirect-argument and bin-count buffers are read as INDIRECT_ARGUMENT;
        // D3D12 implicitly promotes them from COMMON (the graph pins buffers to
        // COMMON and orders the cull->draw write/read with a UAV barrier), so no
        // explicit transition is recorded. The cluster buffers follow the same policy
        // use. Cluster buffers likewise decay to COMMON after this command
        // list's ExecuteCommandLists, keeping the graphics->compute (next frame)
        // handoff legal without asserting a cross-queue before-state.
        DX12GraphResourceEntry* indirectArgsEntry = gpuDrivenBufferEntry(
            plan, GraphResourceSemantic::GpuDrivenIndirectArguments);
        DX12GraphResourceEntry* binCountsEntry = gpuDrivenBufferEntry(
            plan, GraphResourceSemantic::GpuDrivenBinCounts);
        ID3D12CommandSignature* indirectSignature =
            m_gpuScene.indirectCommandSignature(pipeline->rootSignature.Get());
        const bool useGpuDriven =
            indirectSignature &&
            indirectArgsEntry && indirectArgsEntry->buffer &&
            binCountsEntry && binCountsEntry->buffer &&
            !m_gpuScene.geometryBins().empty();

        DX12ForwardSceneInputs sceneInputs{};
        sceneInputs.pipeline = pipeline;
        sceneInputs.usesClusterData = usesClusterData;
        sceneInputs.frameConstantsAddr =
            frameResources.frameConstants.gpuAddress;
        sceneInputs.objectSrv = frameResources.objectSrv.gpuStart;
        sceneInputs.materialSrv = frameResources.materialSrv.gpuStart;
        sceneInputs.iblBaked = m_environmentResources.iblBaked;
        sceneInputs.irradianceSrv =
            m_environmentResources.irradianceSrv.gpuStart;
        sceneInputs.prefilteredSrv =
            m_environmentResources.prefilteredSrv.gpuStart;
        sceneInputs.brdfLutSrv = m_environmentResources.brdfLutSrv.gpuStart;
        sceneInputs.diffuseGiSrv = sceneInputs.brdfLutSrv;
        if (DX12GraphResourceEntry* gi =
                m_graphResourceRegistry.entry(m_resolvedDiffuseGi))
        {
            if (gi->fullSrv.valid())
            {
                sceneInputs.diffuseGiSrv = gi->fullSrv.gpuStart;
            }
            else if (!gi->mipSrvs.empty())
            {
                sceneInputs.diffuseGiSrv = gi->mipSrvs[0].gpuStart;
            }
        }
        sceneInputs.environmentSampler =
            m_environmentResources.sampler.gpuStart;
        sceneInputs.useGpuDriven = useGpuDriven;
        sceneInputs.indirectStream.commandSignature = indirectSignature;
        sceneInputs.indirectStream.indirectArguments = useGpuDriven
            ? indirectArgsEntry->buffer.resource.Get() : nullptr;
        sceneInputs.indirectStream.binCounts = useGpuDriven
            ? binCountsEntry->buffer.resource.Get() : nullptr;
        sceneInputs.indirectStream.commandStride =
            sizeof(DX12GpuIndexedIndirectCommand);
        sceneInputs.draws = m_gpuScene.draws();
        sceneInputs.geometryBins = m_gpuScene.geometryBins();
        sceneInputs.resolveNativeModel =
            [this](AssetHandle handle) -> DX12UploadedModel*
            {
                auto it = m_uploadedModels.find(handle);
                return it != m_uploadedModels.end() ? &it->second : nullptr;
            };

        DX12PassContext passCtx =
            makePassContext(plan, node, ctx, scene, cmd, swapchainImage);
        recordForwardScene(passCtx, sceneInputs);
    }

    void DX12Backend::executeComputeNode(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node,
        [[maybe_unused]] const FrameContext& ctx,
        [[maybe_unused]] const SceneRenderView& scene,
        [[maybe_unused]] ID3D12GraphicsCommandList4* cmd)
    {
        // dispatchNode's std::visit routes PathTrace/Tonemap/EnvironmentConvert
        // payloads to their own recorders, so this node is now only reached for a
        // generic ComputePassData; the per-layout decision below is owned here.
        const ComputePassData* pass =
            std::get_if<ComputePassData>(&plan.payloads[node.payloadIndex]);
        if (!pass || !pass->pipeline)
        {
            return;
        }

        DX12ComputePipeline* pipeline =
            m_pipelineManager.computePipeline(
                computePipelineForNode(plan, node));
        if (!pipeline)
        {
            return;
        }

        const bool measureCull =
            pipeline->desc.bindingLayout ==
                PipelineBindingLayoutKind::GpuFrustumCull &&
            m_gpuCullDebugMode != GpuCullDebugMode::Off;
        const auto cullRecordStart = std::chrono::steady_clock::now();

        cmd->SetComputeRootSignature(pipeline->rootSignature.Get());
        cmd->SetPipelineState(pipeline->pipelineState.Get());

        if (pipeline->desc.bindingLayout ==
            PipelineBindingLayoutKind::ComputeStorageBuffer)
        {
            ensureComputeTestBuffer();
            if (m_computeTestBufferState !=
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            {
                transitionResource(
                    cmd,
                    m_computeTestBuffer.resource.Get(),
                    m_computeTestBufferState,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                m_computeTestBufferState =
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            }

            DX12PassContext passCtx{};
            passCtx.cmd = cmd;
            DX12ComputeStorageBufferTest test{};
            test.bufferAddr = m_computeTestBuffer.gpuAddress;
            test.bufferResource = m_computeTestBuffer.resource.Get();
            test.groupCountX = pass->groupCountX;
            test.groupCountY = pass->groupCountY;
            test.groupCountZ = pass->groupCountZ;
            recordComputeStorageBufferTest(passCtx, test);
            return;
        }
        else if (pipeline->desc.bindingLayout ==
            PipelineBindingLayoutKind::GlobalIllumination)
        {
            // Cache initialization must execute exactly once as soon as graph
            // resources exist, even if streamed RT scene descriptors are not
            // ready yet. The shader only references the two root UAVs below.
            if (pass->name == "GI.CacheInitialization")
            {
                m_giCacheInitializationFrame = ctx.frameIndex;
                m_giDiagnosticFrames = 0;
                m_loggedGiDiagnostics = false;
                DX12GraphResourceEntry* hash = m_graphResourceRegistry.entry(
                    findResourceBySemantic(
                        plan, GraphResourceSemantic::GiHashBuckets));
                DX12GraphResourceEntry* residual =
                    m_graphResourceRegistry.entry(findResourceBySemantic(
                        plan, GraphResourceSemantic::GiResidualInterface));
                DX12GraphResourceEntry* probes =
                    m_graphResourceRegistry.entry(findResourceBySemantic(
                        plan, GraphResourceSemantic::GiProbes));
                DX12GraphResourceEntry* probeStaging =
                    m_graphResourceRegistry.entry(findResourceBySemantic(
                        plan, GraphResourceSemantic::GiProbeStaging));
                if (!hash || !hash->buffer || !residual || !residual->buffer ||
                    !probes || !probes->buffer || !probeStaging ||
                    !probeStaging->buffer)
                    return;
                cmd->SetComputeRootUnorderedAccessView(
                    3u, hash->buffer.gpuAddress);
                cmd->SetComputeRootUnorderedAccessView(
                    10u, residual->buffer.gpuAddress);
                cmd->SetComputeRootUnorderedAccessView(
                    4u, probes->buffer.gpuAddress);
                cmd->SetComputeRootUnorderedAccessView(
                    12u, probeStaging->buffer.gpuAddress);
                cmd->Dispatch(pass->groupCountX,
                    pass->groupCountY, pass->groupCountZ);
                return;
            }
            if (m_gpuScene.frameSlotCount() == 0 ||
                !m_rayTracingSceneService)
                return;
            const uint32_t frameSlot = static_cast<uint32_t>(
                ctx.frameIndex % m_gpuScene.frameSlotCount());
            DX12GpuSceneFrameResources& frameResources =
                m_gpuScene.frameResources(frameSlot);
            if (!frameResources.giRtGeometries ||
                !frameResources.giRtInstances ||
                !frameResources.materials ||
                !frameResources.giTraceConstants ||
                !frameResources.frameConstants ||
                !frameResources.giModelBufferSrvs.valid())
                return;
            ID3D12DescriptorHeap* heaps[] = {
                m_descriptorSystem.shaderResourceHeap(),
                m_descriptorSystem.samplerHeap() };
            cmd->SetDescriptorHeaps(2, heaps);
            for (uint32_t i = 0; i < GiBufferBindingCount; ++i)
            {
                const GraphResourceId id =
                    findResourceBySemantic(plan, GiBufferSemantics[i]);
                DX12GraphResourceEntry* entry =
                    m_graphResourceRegistry.entry(id);
                if (!entry || !entry->buffer)
                    return;
                cmd->SetComputeRootUnorderedAccessView(
                    i, entry->buffer.gpuAddress);
            }

            const auto srvHandle = [](DX12GraphResourceEntry* entry)
            {
                if (!entry)
                    return D3D12_GPU_DESCRIPTOR_HANDLE{};
                if (entry->fullSrv.valid())
                    return entry->fullSrv.gpuStart;
                return !entry->mipSrvs.empty()
                    ? entry->mipSrvs[0].gpuStart
                    : D3D12_GPU_DESCRIPTOR_HANDLE{};
            };
            const GraphResourceId primaryId =
                findNodeResource(plan, node, ResourceUsage::SampledTexture);
            const D3D12_GPU_DESCRIPTOR_HANDLE primary =
                srvHandle(m_graphResourceRegistry.entry(primaryId));
            if (primary.ptr != 0)
                cmd->SetComputeRootDescriptorTable(
                    GiBufferBindingCount, primary);

            const GraphResourceId historyId = findPreviousNodeResource(
                plan, node, ResourceUsage::SampledTexture);
            const D3D12_GPU_DESCRIPTOR_HANDLE history = srvHandle(
                m_graphResourceRegistry.previousEntry(historyId));
            if (history.ptr != 0)
                cmd->SetComputeRootDescriptorTable(
                    GiBufferBindingCount + 1u, history);

            const GraphResourceId sceneDepthId = findResourceBySemantic(
                plan, GraphResourceSemantic::GiSceneDepth);
            const GraphResourceId surfaceAttributesId = findResourceBySemantic(
                plan, GraphResourceSemantic::GiSurfaceAttributes);
            const D3D12_GPU_DESCRIPTOR_HANDLE sceneDepth = srvHandle(
                m_graphResourceRegistry.entry(sceneDepthId));
            const D3D12_GPU_DESCRIPTOR_HANDLE surfaceAttributes = srvHandle(
                m_graphResourceRegistry.entry(surfaceAttributesId));
            const D3D12_GPU_DESCRIPTOR_HANDLE previousSceneDepth = srvHandle(
                m_graphResourceRegistry.previousEntry(sceneDepthId));
            const D3D12_GPU_DESCRIPTOR_HANDLE previousSurfaceAttributes = srvHandle(
                m_graphResourceRegistry.previousEntry(surfaceAttributesId));
            if (sceneDepth.ptr != 0)
                cmd->SetComputeRootDescriptorTable(
                    GiBufferBindingCount + 2u, sceneDepth);
            if (surfaceAttributes.ptr != 0)
                cmd->SetComputeRootDescriptorTable(
                    GiBufferBindingCount + 3u, surfaceAttributes);
            if (previousSceneDepth.ptr != 0)
                cmd->SetComputeRootDescriptorTable(
                    GiBufferBindingCount + 4u, previousSceneDepth);
            if (previousSurfaceAttributes.ptr != 0)
                cmd->SetComputeRootDescriptorTable(
                    GiBufferBindingCount + 5u, previousSurfaceAttributes);

            const GraphResourceId outputId =
                findNodeResource(plan, node, ResourceUsage::StorageTexture);
            DX12GraphResourceEntry* output =
                m_graphResourceRegistry.entry(outputId);
            if (output && !output->mipUavs.empty())
                cmd->SetComputeRootDescriptorTable(
                    GiBufferBindingCount + 6u,
                    output->mipUavs[0].gpuStart);
            const uint64_t tlas =
                m_accelerationStructures.shaderTlasHandle();
            if (tlas == 0) return;
            cmd->SetComputeRootShaderResourceView(
                GiBufferBindingCount + 7u, tlas);
            cmd->SetComputeRootShaderResourceView(
                GiBufferBindingCount + 8u,
                frameResources.giRtGeometries.gpuAddress);
            cmd->SetComputeRootShaderResourceView(
                GiBufferBindingCount + 9u,
                frameResources.giRtInstances.gpuAddress);
            cmd->SetComputeRootShaderResourceView(
                GiBufferBindingCount + 10u,
                frameResources.materials.gpuAddress);
            cmd->SetComputeRootDescriptorTable(
                GiBufferBindingCount + 11u,
                frameResources.giModelBufferSrvs.gpuStart);
            D3D12_GPU_DESCRIPTOR_HANDLE indexBuffers =
                frameResources.giModelBufferSrvs.gpuStart;
            indexBuffers.ptr += static_cast<UINT64>(GiMaxGeometryBufferCount) *
                frameResources.giModelBufferSrvs.descriptorSize;
            cmd->SetComputeRootDescriptorTable(
                GiBufferBindingCount + 12u, indexBuffers);
            cmd->SetComputeRootDescriptorTable(
                GiBufferBindingCount + 13u,
                m_descriptorSystem.shaderResourceGpuStart());
            cmd->SetComputeRootDescriptorTable(
                GiBufferBindingCount + 14u,
                m_descriptorSystem.samplerGpuStart());
            D3D12_GPU_DESCRIPTOR_HANDLE environment =
                m_environmentResources.cubemapSrv.gpuStart;
            const bool environmentValid =
                m_environmentResources.cubemapSrv.valid();
            if (!environmentValid)
            {
                environment = frameResources.giModelBufferSrvs.gpuStart;
                environment.ptr += static_cast<UINT64>(
                    GiMaxGeometryBufferCount * 2u) *
                    frameResources.giModelBufferSrvs.descriptorSize;
            }
            cmd->SetComputeRootDescriptorTable(
                GiBufferBindingCount + 15u, environment);
            cmd->SetComputeRootDescriptorTable(
                GiBufferBindingCount + 16u,
                m_environmentResources.sampler.valid()
                    ? m_environmentResources.sampler.gpuStart
                    : m_descriptorSystem.samplerGpuStart());
            const auto& rtStats = m_rayTracingSceneService->statistics();
            float cellSize = 0.25f;
            std::memcpy(&cellSize, &pass->userData[6], sizeof(float));
            cellSize = std::max(cellSize, 1.0e-3f);
            const uint32_t hierarchyBits = pass->userData[7];
            const uint32_t evaluationDivisor = std::max(
                hierarchyBits & 0xfu, 1u);
            const uint32_t clipmapCount = std::clamp(
                (hierarchyBits >> 4u) & 0xfu, 1u, 8u);
            const uint32_t probeResolution = std::clamp(
                (hierarchyBits >> 8u) & 0x7fu, 2u, 64u);
            GpuGiTraceConstants constants{
                .frameIndex = static_cast<uint32_t>(ctx.frameIndex),
                .sceneGeneration = static_cast<uint32_t>(rtStats.generation),
                .geometryCount = static_cast<uint32_t>(
                    m_rayTracingSceneService->gpuGeometries().size()),
                .instanceCount = static_cast<uint32_t>(
                    m_rayTracingSceneService->gpuInstances().size()),
                .materialCount = frameResources.materialCapacity,
                .textureCount = MaxBindlessTextures,
                .samplerCount = MaxBindlessSamplers,
                .maxUpdates = m_giMaxSurfelUpdates,
                .rayBudget = m_giRayBudget,
                .raysPerSurfel = std::max(pass->userData[2], 1u),
                .debugView = m_giDebugView,
                .environmentEnabled = environmentValid ? 1u : 0u,
                // Drive the probe-transport sky from the scene so it is
                // controllable and consistent with the forward/PT paths (it was
                // defaulting to 1.0, ignoring the scene environment intensity).
                .environmentIntensity =
                    scene.environment.settings.intensity,
                .maxSurfels = pass->userData[4],
                .cellSize = cellSize,
                .invCellSize = 1.0f / cellSize,
                .hashBucketCount = pass->userData[5],
                .candidatesPerCell = 4u,
                .gatherRadiusScale = 1.5f,
                .normalThreshold = 0.85f,
                .planeThreshold = 0.6f * cellSize,
                .maxSurfelAge = 256u,
                .reducedWidth = (m_swapchain.width() +
                    evaluationDivisor - 1u) / evaluationDivisor,
                .reducedHeight = (m_swapchain.height() +
                    evaluationDivisor - 1u) / evaluationDivisor,
                .evaluationDivisor = evaluationDivisor,
                .feedbackEnabled = (hierarchyBits >> 15u) & 1u,
                .confidenceBlend = static_cast<float>(m_giMaxProbeUpdates),
                .emissiveInstanceIndex =
                    rtStats.firstEmissiveInstanceIndex,
                .giFlags = (m_giDiagnosticsReadbackActive ? 1u : 0u) |
                    (clipmapCount << 4u) |
                    (probeResolution << 8u) |
                    ((hierarchyBits & (1u << 16u)) != 0u
                        ? (1u << 20u) : 0u) |
                    ((hierarchyBits & (1u << 17u)) != 0u
                        ? (1u << 21u) : 0u),
                .freezeAfterFrame = m_giFreezeAfterFrames == 0u ? 0u :
                    static_cast<uint32_t>(m_giCacheInitializationFrame +
                        m_giFreezeAfterFrames) };
            if (frameResources.giTraceConstants.mapped)
                std::memcpy(frameResources.giTraceConstants.mapped,
                    &constants, sizeof(constants));
            cmd->SetComputeRootConstantBufferView(
                GiBufferBindingCount + 17u,
                frameResources.giTraceConstants.gpuAddress);
            cmd->SetComputeRootConstantBufferView(
                GiBufferBindingCount + 18u,
                frameResources.frameConstants.gpuAddress);

            if (pass->indirectArguments != InvalidGraphResourceId)
            {
                DX12GraphResourceEntry* arguments =
                    m_graphResourceRegistry.entry(pass->indirectArguments);
                if (!arguments || !arguments->buffer ||
                    !m_dispatchIndirectSignature)
                {
                    return;
                }
                cmd->ExecuteIndirect(
                    m_dispatchIndirectSignature.Get(),
                    1,
                    arguments->buffer.resource.Get(),
                    pass->indirectArgumentOffset,
                    nullptr,
                    0);
            }
            else
            {
                cmd->Dispatch(pass->groupCountX,
                    pass->groupCountY, pass->groupCountZ);
            }
            return;
        }
        else if (pipeline->desc.bindingLayout ==
            PipelineBindingLayoutKind::HiZDepthPyramid)
        {
            // Depth target is ensured serially before recording (executeGraph).
            if (m_gpuScene.frameSlotCount() == 0)
            {
                return;
            }
            const uint32_t frameSlot = static_cast<uint32_t>(
                ctx.frameIndex % m_gpuScene.frameSlotCount());
            DX12GpuSceneFrameResources& frameResources =
                m_gpuScene.frameResources(frameSlot);

            DX12PassContext passCtx{};
            passCtx.cmd = cmd;
            passCtx.plan = &plan;
            passCtx.node = &node;
            passCtx.resources = &m_graphResourceRegistry;
            passCtx.descriptors = &m_descriptorSystem;

            DX12HiZInputs hiZ{};
            hiZ.hiZId =
                findNodeResource(plan, node, ResourceUsage::StorageTexture);
            hiZ.sceneDepthId =
                findNodeResource(plan, node, ResourceUsage::SampledTexture);
            hiZ.frameConstantsAddr = frameResources.frameConstants
                ? frameResources.frameConstants.gpuAddress
                : 0;
            hiZ.hiZDebugResourceOut =
                &m_clusteredForwardResources.hiZDebugResource;

            (void)recordHiZPyramid(passCtx, *pipeline, hiZ);
            return;
        }
        else if (pipeline->desc.bindingLayout ==
            PipelineBindingLayoutKind::GpuFrustumCull)
        {
            ensureClusteredForwardResources();
            if (!prepareSceneResources(ctx, scene) ||
                m_gpuScene.frameSlotCount() == 0)
            {
                return;
            }
            const uint32_t frameSlot = static_cast<uint32_t>(
                ctx.frameIndex % m_gpuScene.frameSlotCount());
            DX12GpuSceneFrameResources& frameResources =
                m_gpuScene.frameResources(frameSlot);
            DX12GpuScene& g = m_gpuScene;

            DX12PassContext passCtx{};
            passCtx.cmd = cmd;
            passCtx.plan = &plan;
            passCtx.node = &node;
            passCtx.resources = &m_graphResourceRegistry;
            passCtx.descriptors = &m_descriptorSystem;

            // Resolve every cull buffer from the graph registry (per frame
            // slot). The graph owns their state and cross-pass ordering; the
            // recorder only binds addresses.
            DX12GraphResourceEntry* visibleInstances = gpuDrivenBufferEntry(
                plan, GraphResourceSemantic::GpuDrivenVisibleInstances);
            DX12GraphResourceEntry* visibleCount = gpuDrivenBufferEntry(
                plan, GraphResourceSemantic::GpuDrivenVisibleCount);
            DX12GraphResourceEntry* indirectArgs = gpuDrivenBufferEntry(
                plan, GraphResourceSemantic::GpuDrivenIndirectArguments);
            DX12GraphResourceEntry* binCounts = gpuDrivenBufferEntry(
                plan, GraphResourceSemantic::GpuDrivenBinCounts);
            DX12GraphResourceEntry* instanceBounds = gpuDrivenBufferEntry(
                plan, GraphResourceSemantic::GpuDrivenInstanceBounds);
            DX12GraphResourceEntry* drawInputs = gpuDrivenBufferEntry(
                plan, GraphResourceSemantic::GpuDrivenDrawInputs);
            DX12GraphResourceEntry* cullClassification =
                gpuDrivenBufferEntry(
                    plan,
                    GraphResourceSemantic::GpuDrivenCullClassification);
            DX12GraphResourceEntry* cullStats = gpuDrivenBufferEntry(
                plan, GraphResourceSemantic::GpuDrivenCullStats);
            const GraphResourceId previousHiZId =
                findPreviousNodeResource(
                    plan, node, ResourceUsage::SampledTexture);
            DX12GraphResourceEntry* previousHiZ =
                previousHiZId != InvalidGraphResourceId
                    ? m_graphResourceRegistry.previousEntry(previousHiZId)
                    : nullptr;
            if (!visibleInstances || !visibleCount || !indirectArgs ||
                !binCounts || !instanceBounds || !drawInputs ||
                !cullClassification || !cullStats ||
                (previousHiZId != InvalidGraphResourceId &&
                    (!previousHiZ || !previousHiZ->fullSrv.valid())))
            {
                return;
            }

            DX12CullBuffers cull{};
            cull.visibleInstancesAddr = visibleInstances->buffer.gpuAddress;
            cull.visibleInstanceCountAddr = visibleCount->buffer.gpuAddress;
            cull.indirectArgumentsAddr = indirectArgs->buffer.gpuAddress;
            cull.binCountsAddr = binCounts->buffer.gpuAddress;
            cull.frameConstantsAddr = frameResources.frameConstants.gpuAddress;
            cull.instanceBoundsAddr = instanceBounds->buffer.gpuAddress;
            cull.drawInputsAddr = drawInputs->buffer.gpuAddress;
            cull.cullClassificationAddr =
                cullClassification->buffer.gpuAddress;
            cull.cullStatsAddr = cullStats->buffer.gpuAddress;
            if (previousHiZ)
            {
                cull.previousHiZSrv = previousHiZ->fullSrv.gpuStart;
            }

            recordGpuFrustumCull(passCtx, cull);

            if (!g.loggedGpuCull)
            {
                spdlog::info(
                    "[DX12Backend] GPU frustum + previous Hi-Z occlusion culling prepared for {} instance(s)",
                    m_gpuScene.instanceCount());
                g.loggedGpuCull = true;
            }
        }
        else if (pipeline->desc.bindingLayout ==
            PipelineBindingLayoutKind::GpuOcclusionValidation)
        {
            if (m_gpuScene.frameSlotCount() == 0)
            {
                return;
            }
            const uint32_t frameSlot = static_cast<uint32_t>(
                ctx.frameIndex % m_gpuScene.frameSlotCount());
            DX12GpuSceneFrameResources& frameResources =
                m_gpuScene.frameResources(frameSlot);
            DX12GraphResourceEntry* bounds = gpuDrivenBufferEntry(
                plan, GraphResourceSemantic::GpuDrivenInstanceBounds);
            DX12GraphResourceEntry* classification = gpuDrivenBufferEntry(
                plan,
                GraphResourceSemantic::GpuDrivenCullClassification);
            DX12GraphResourceEntry* stats = gpuDrivenBufferEntry(
                plan, GraphResourceSemantic::GpuDrivenCullStats);
            const GraphResourceId hiZId = findNodeResource(
                plan, node, ResourceUsage::SampledTexture);
            DX12GraphResourceEntry* hiZ =
                m_graphResourceRegistry.entry(hiZId);
            if (!bounds || !classification || !stats || !hiZ ||
                !hiZ->fullSrv.valid())
            {
                return;
            }
            DX12PassContext passCtx{};
            passCtx.cmd = cmd;
            passCtx.descriptors = &m_descriptorSystem;
            DX12OcclusionValidationInputs validation{};
            validation.frameConstantsAddr =
                frameResources.frameConstants.gpuAddress;
            validation.instanceBoundsAddr = bounds->buffer.gpuAddress;
            validation.cullClassificationAddr =
                classification->buffer.gpuAddress;
            validation.cullStatsAddr = stats->buffer.gpuAddress;
            validation.currentHiZSrv = hiZ->fullSrv.gpuStart;
            recordGpuOcclusionValidation(passCtx, validation);
        }
        else if (pipeline->desc.bindingLayout ==
            PipelineBindingLayoutKind::ClusteredForward)
        {
            if (!bindClusteredForwardCompute(*pipeline, plan, ctx, scene, cmd))
            {
                return;
            }
        }

        if (measureCull)
        {
            const uint32_t frameSlot = static_cast<uint32_t>(
                ctx.frameIndex % m_frameExecutor.framesInFlight());
            cmd->EndQuery(
                m_gpuCullTimestampHeap.Get(),
                D3D12_QUERY_TYPE_TIMESTAMP,
                frameSlot * 2u);
        }

        if (pipeline->desc.bindingLayout ==
                PipelineBindingLayoutKind::GpuFrustumCull &&
            nodeUsesResourceSemantic(
                plan,
                node,
                GraphResourceSemantic::GpuDrivenInstanceBounds))
        {
            const uint32_t instanceCount =
                std::min<uint32_t>(
                    m_gpuScene.instanceCount(),
                    ClusteredForwardMaxGpuCullInstances);
            cmd->Dispatch(
                std::max(1u, (instanceCount + 63u) / 64u),
                1,
                1);
        }
        else
        {
            cmd->Dispatch(
                pass->groupCountX,
                pass->groupCountY,
                pass->groupCountZ);
        }

        if (measureCull)
        {
            const uint32_t frameSlot = static_cast<uint32_t>(
                ctx.frameIndex % m_frameExecutor.framesInFlight());
            cmd->EndQuery(
                m_gpuCullTimestampHeap.Get(),
                D3D12_QUERY_TYPE_TIMESTAMP,
                frameSlot * 2u + 1u);
            cmd->ResolveQueryData(
                m_gpuCullTimestampHeap.Get(),
                D3D12_QUERY_TYPE_TIMESTAMP,
                frameSlot * 2u,
                2u,
                m_gpuCullTimestampReadback.resource.Get(),
                sizeof(uint64_t) * frameSlot * 2u);
            m_gpuCullCpuRecordMilliseconds[frameSlot] =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - cullRecordStart).count();
            m_gpuCullTimestampValid[frameSlot] = 1u;
        }

        // The GPU-driven cull buffers (and the clustered cluster buffers) are
        // frame-graph owned: the graph emits the write->read barriers from the
        // declared dependencies and D3D12 implicit buffer promotion/decay carries
        // the state across the cull -> indirect-draw handoff, so no manual
        // post-dispatch barrier or state transition is recorded here.
    }

    void DX12Backend::executeEnvironmentConvertNode(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node,
        const FrameContext& ctx,
        const SceneRenderView& scene,
        ID3D12GraphicsCommandList4* cmd)
    {
        if (!ensureEnvironmentResources(ctx, scene, cmd) ||
            m_environmentResources.converted)
        {
            return;
        }

        DX12ComputePipeline* pipeline =
            m_pipelineManager.computePipeline(
                computePipelineForNode(plan, node));
        if (!pipeline)
        {
            return;
        }

        (void)convertEnvironmentIfReady(*pipeline, ctx, scene, cmd);
    }

    DX12ComputePipeline* DX12Backend::environmentConvertPipeline()
    {
        if (!m_pipelineLibrary)
        {
            return nullptr;
        }

        const PipelineId pipelineId = makePipelineId("equirect_to_cubemap");
        const ComputePipelineHandle handle =
            m_pipelineManager.resolveComputePipeline(
                *m_pipelineLibrary, pipelineId);
        return m_pipelineManager.computePipeline(handle);
    }

    bool DX12Backend::convertEnvironmentIfReady(
        DX12ComputePipeline& pipeline,
        const FrameContext& ctx,
        const SceneRenderView& scene,
        ID3D12GraphicsCommandList4* cmd)
    {
        if (!ensureEnvironmentResources(ctx, scene, cmd) ||
            m_environmentResources.converted)
        {
            return m_environmentResources.converted;
        }
        if (!pipeline.rootSignature || !pipeline.pipelineState)
        {
            spdlog::error(
                "[DX12] Environment conversion pipeline is incomplete; skipping HDR conversion");
            return false;
        }

        const uint64_t key =
            uploadedTextureKey(
                scene.environment.equirectTexture,
                0,
                TextureTransferFunction::Linear);
        auto textureIt = m_uploadedTextures.find(key);
        if (textureIt == m_uploadedTextures.end())
        {
            return false;
        }

        if (m_environmentResources.cubemapState !=
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        {
            transitionResource(
                cmd,
                m_environmentResources.cubemap.resource.Get(),
                m_environmentResources.cubemapState,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            m_environmentResources.cubemapState =
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }

        DX12PassContext passCtx{};
        passCtx.cmd = cmd;
        passCtx.descriptors = &m_descriptorSystem;

        DX12EnvironmentConvertInputs convertInputs{};
        convertInputs.pipeline = &pipeline;
        convertInputs.sourceSrv = textureIt->second.srv.gpuStart;
        convertInputs.cubemapUav = m_environmentResources.cubemapUav.gpuStart;
        convertInputs.sampler = m_environmentResources.sampler.gpuStart;
        convertInputs.cubemap = m_environmentResources.cubemap.resource.Get();
        convertInputs.cubemapSize = m_environmentResources.cubemapSize;
        if (!recordEnvironmentConvert(passCtx, convertInputs))
        {
            return false;
        }

        transitionResource(
            cmd,
            m_environmentResources.cubemap.resource.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        m_environmentResources.cubemapState =
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        m_environmentResources.converted = true;
        m_pathTraceResources.resetAccumulation = true;
        spdlog::info("[DX12] Converted HDR environment to cubemap");
        return true;
    }

    void DX12Backend::executePathTraceNode(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node,
        const FrameContext& ctx,
        const SceneRenderView& scene,
        ID3D12GraphicsCommandList4* cmd)
    {
        const ComputePipelineHandle pathTracePipelineHandle =
            computePipelineForNode(plan, node);

        ensurePathTraceResources();
        ensurePathTraceSceneResources(ctx, scene);
        if (!m_pathTraceResources.accumulation)
        {
            return;
        }

        const bool hasCamera = scene.camera.valid != 0u;
        const bool cameraChanged =
            hasCamera &&
            (!m_pathTraceResources.hasPreviousCamera ||
                matricesDiffer(scene.camera.view, m_pathTraceResources.previousView) ||
                matricesDiffer(
                    scene.camera.projection,
                    m_pathTraceResources.previousProjection));

        if (cameraChanged)
        {
            m_pathTraceResources.accumulatedSampleCount = 0;
            m_pathTraceResources.resetAccumulation = true;
            m_pathTraceResources.previousView = scene.camera.view;
            m_pathTraceResources.previousProjection = scene.camera.projection;
            m_pathTraceResources.hasPreviousCamera = true;
        }
        else if (!hasCamera)
        {
            m_pathTraceResources.hasPreviousCamera = false;
        }

        if (m_pathTraceResources.environmentVersion !=
            scene.environment.version)
        {
            m_pathTraceResources.environmentVersion =
                scene.environment.version;
            m_pathTraceResources.accumulatedSampleCount = 0;
            m_pathTraceResources.resetAccumulation = true;
        }
        m_pathTraceResources.tonemapExposure =
            scene.environment.settings.tonemapExposure;

        if (m_pathTraceResources.accumulationState !=
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        {
            transitionResource(
                cmd,
                m_pathTraceResources.accumulation.resource.Get(),
                m_pathTraceResources.accumulationState,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            m_pathTraceResources.accumulationState =
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }

        // Environment lifetime/conversion is the backend's job; hoisting it here
        // (ahead of the CPU-only constants build the recorder now owns) leaves
        // the RECORDED command order identical: accumulation transition, then any
        // environment-convert dispatch, then the path-trace dispatch.
        const bool environmentRequested =
            scene.environment.enabled != 0u &&
            static_cast<bool>(scene.environment.equirectTexture);
        const bool environmentResourcesAvailable = environmentRequested &&
            ensureEnvironmentResources(ctx, scene, cmd);
        if (environmentResourcesAvailable && !m_environmentResources.converted)
        {
            if (DX12ComputePipeline* convertPipeline =
                    environmentConvertPipeline())
            {
                (void)convertEnvironmentIfReady(
                    *convertPipeline,
                    ctx,
                    scene,
                    cmd);
            }
        }
        const bool environmentReady = environmentResourcesAvailable &&
            m_environmentResources.converted;

        DX12ComputePipeline* pipeline =
            m_pipelineManager.computePipeline(pathTracePipelineHandle);
        if (!pipeline)
        {
            return;
        }

        // Patch the environment cubemap SRV into the scene descriptor table.
        // Resource-view creation is backend resolution, not command recording.
        if (m_pathTraceResources.sceneSrvs.valid() &&
            m_environmentResources.cubemapSrv.valid() &&
            m_environmentResources.cubemap.resource &&
            m_pathTraceResources.sceneSrvs.count > 4u)
        {
            D3D12_CPU_DESCRIPTOR_HANDLE environmentDst =
                m_pathTraceResources.sceneSrvs.cpuStart;
            environmentDst.ptr +=
                static_cast<SIZE_T>(4u) *
                m_pathTraceResources.sceneSrvs.descriptorSize;

            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.Shader4ComponentMapping =
                D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.Format = m_environmentResources.cubemap.desc.Format;
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
            srv.TextureCube.MipLevels = 1;
            m_device.device()->CreateShaderResourceView(
                m_environmentResources.cubemap.resource.Get(),
                &srv,
                environmentDst);
        }

        const uint32_t frameSlot =
            static_cast<uint32_t>(
                ctx.frameIndex %
                m_pathTraceResources.pathTraceConstants.size());
        DX12Buffer& pathTraceConstants =
            m_pathTraceResources.pathTraceConstants[frameSlot];

        DX12PathTraceInputs inputs{};
        inputs.pipeline = pipeline;
        inputs.constantsMapped = pathTraceConstants.mapped;
        inputs.constantsAddr = pathTraceConstants.gpuAddress;
        inputs.accumulationUav = m_pathTraceResources.accumulationUav.gpuStart;
        inputs.sceneSrvsValid = m_pathTraceResources.sceneSrvs.valid();
        inputs.sceneSrvs = m_pathTraceResources.sceneSrvs.gpuStart;
        inputs.samplerValid = m_environmentResources.sampler.valid();
        inputs.sampler = m_environmentResources.sampler.gpuStart;
        inputs.accumulation = m_pathTraceResources.accumulation.resource.Get();
        inputs.width = m_pathTraceResources.width;
        inputs.height = m_pathTraceResources.height;
        inputs.frameIndex = static_cast<uint32_t>(ctx.frameIndex);
        inputs.accumulatedSampleCount =
            m_pathTraceResources.accumulatedSampleCount;
        inputs.exposure = m_pathTraceResources.tonemapExposure;
        inputs.resetAccumulation = m_pathTraceResources.resetAccumulation;
        inputs.sceneVertexCount = m_pathTraceResources.sceneVertexCount;
        inputs.sceneMaterialCount = m_pathTraceResources.sceneMaterialCount;
        inputs.sceneTriangleCount = m_pathTraceResources.sceneTriangleCount;
        inputs.sceneBvhNodeCount = m_pathTraceResources.sceneBvhNodeCount;
        inputs.firstEmissiveTriangleIndex =
            m_pathTraceResources.firstEmissiveTriangleIndex;
        inputs.emissiveTriangleCount =
            m_pathTraceResources.emissiveTriangleCount;
        inputs.environmentReady = environmentReady;
        inputs.environmentIntensity = scene.environment.settings.intensity;
        inputs.environmentExposure =
            scene.environment.settings.pathTraceExposure;

        DX12PassContext passCtx =
            makePassContext(plan, node, ctx, scene, cmd, nullptr);
        if (!recordPathTrace(passCtx, inputs))
        {
            return;
        }

        ++m_pathTraceResources.accumulatedSampleCount;
        m_pathTraceResources.resetAccumulation = false;
    }

    void DX12Backend::executeTonemapNode(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node,
        const FrameContext& ctx,
        ID3D12GraphicsCommandList4* cmd)
    {
        ensurePathTraceResources();

        DX12ComputePipeline* pipeline =
            m_pipelineManager.computePipeline(
                computePipelineForNode(plan, node));
        if (!pipeline ||
            !m_pathTraceResources.accumulation ||
            !m_pathTraceResources.tonemap)
        {
            return;
        }

        if (m_pathTraceResources.accumulationState !=
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
        {
            transitionResource(
                cmd,
                m_pathTraceResources.accumulation.resource.Get(),
                m_pathTraceResources.accumulationState,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            m_pathTraceResources.accumulationState =
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        }

        if (m_pathTraceResources.tonemapState !=
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        {
            transitionResource(
                cmd,
                m_pathTraceResources.tonemap.resource.Get(),
                m_pathTraceResources.tonemapState,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            m_pathTraceResources.tonemapState =
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }

        const uint32_t frameSlot =
            static_cast<uint32_t>(
                ctx.frameIndex %
                m_pathTraceResources.tonemapConstants.size());
        DX12Buffer& tonemapConstants =
            m_pathTraceResources.tonemapConstants[frameSlot];

        DX12TonemapInputs inputs{};
        inputs.pipeline = pipeline;
        inputs.constantsMapped = tonemapConstants.mapped;
        inputs.constantsAddr = tonemapConstants.gpuAddress;
        inputs.tonemapUav = m_pathTraceResources.tonemapUav.gpuStart;
        inputs.accumulationSrv = m_pathTraceResources.accumulationSrv.gpuStart;
        inputs.tonemap = m_pathTraceResources.tonemap.resource.Get();
        inputs.width = m_pathTraceResources.width;
        inputs.height = m_pathTraceResources.height;
        inputs.exposure = m_pathTraceResources.tonemapExposure;

        DX12PassContext passCtx{};
        passCtx.cmd = cmd;
        passCtx.plan = &plan;
        passCtx.node = &node;
        passCtx.frame = &ctx;
        passCtx.descriptors = &m_descriptorSystem;
        recordTonemap(passCtx, inputs);
    }

    void DX12Backend::executeTransferNode(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node,
        [[maybe_unused]] const FrameContext& ctx,
        ID3D12GraphicsCommandList4* cmd,
        ID3D12Resource* swapchainImage)
    {
        if (node.payloadIndex >= plan.payloads.size())
        {
            return;
        }

        const TransferPassData* pass =
            std::get_if<TransferPassData>(&plan.payloads[node.payloadIndex]);
        if (!pass)
        {
            return;
        }

        if (pass->source >= plan.resources.size() ||
            pass->destination >= plan.resources.size())
        {
            spdlog::error("Transfer pass '{}' has invalid graph resources", pass->name);
            return;
        }

        const GraphResource& sourceDesc = plan.resources[pass->source];
        const GraphResource& destinationDesc = plan.resources[pass->destination];
        auto resolve = [&](const GraphResource& resource) -> ID3D12Resource*
        {
            if (resource.imported == ImportedResource::Swapchain)
            {
                return swapchainImage;
            }
            if (resource.semantic == GraphResourceSemantic::PathTraceTonemap)
            {
                return m_pathTraceResources.tonemap
                    ? m_pathTraceResources.tonemap.resource.Get() : nullptr;
            }
            return m_graphResourceRegistry.nativeResource(resource.id);
        };

        DX12TransferCopy copy{};
        copy.source = resolve(sourceDesc);
        copy.destination = resolve(destinationDesc);
        copy.type = sourceDesc.type;
        copy.passName = pass->name.c_str();
        if (sourceDesc.type != destinationDesc.type)
        {
            spdlog::error("Transfer pass '{}' cannot copy between resource types", pass->name);
            return;
        }

        // Backend owns the state-tracked transition of its private path-trace
        // target; the recorder validates and issues the copy.
        if (sourceDesc.semantic == GraphResourceSemantic::PathTraceTonemap)
        {
            if (m_pathTraceResources.tonemapState !=
                D3D12_RESOURCE_STATE_COPY_SOURCE)
            {
                transitionResource(
                    cmd,
                    m_pathTraceResources.tonemap.resource.Get(),
                    m_pathTraceResources.tonemapState,
                    D3D12_RESOURCE_STATE_COPY_SOURCE);
                m_pathTraceResources.tonemapState =
                    D3D12_RESOURCE_STATE_COPY_SOURCE;
            }
        }

        recordTransferCopy(cmd, copy);
    }

    GraphicsPipelineHandle DX12Backend::pipelineForNode(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node)
    {
        if (!m_pipelineLibrary ||
            node.payloadIndex >= plan.payloads.size())
        {
            return {};
        }

        const GraphicsPassData* pass =
            std::get_if<GraphicsPassData>(&plan.payloads[node.payloadIndex]);
        if (!pass || !pass->pipeline)
        {
            return {};
        }

        return m_pipelineManager.resolveGraphicsPipeline(
            *m_pipelineLibrary, pass->pipeline, swapchainTextureFormat());
    }

    ComputePipelineHandle DX12Backend::computePipelineForNode(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node)
    {
        if (!m_pipelineLibrary ||
            node.payloadIndex >= plan.payloads.size())
        {
            return {};
        }

        PipelineId pipelineId{};

        if (const ComputePassData* computePass =
            std::get_if<ComputePassData>(&plan.payloads[node.payloadIndex]))
        {
            pipelineId = computePass->pipeline;
        }
        else if (const PathTracePassData* pathTracePass =
            std::get_if<PathTracePassData>(&plan.payloads[node.payloadIndex]))
        {
            pipelineId = pathTracePass->pipeline;
        }
        else if (const TonemapPassData* tonemapPass =
            std::get_if<TonemapPassData>(&plan.payloads[node.payloadIndex]))
        {
            pipelineId = tonemapPass->pipeline;
        }

        if (!pipelineId)
        {
            return {};
        }

        return m_pipelineManager.resolveComputePipeline(
            *m_pipelineLibrary, pipelineId);
    }

    void DX12Backend::destroySceneResources()
    {
        destroyDepthTarget();
        destroyPathTraceResources();
        destroyClusteredForwardResources();
        m_resourceAllocator.destroyBuffer(m_computeTestBuffer);
        m_computeTestBufferState = D3D12_RESOURCE_STATE_COMMON;

        for (auto& [handle, model] : m_uploadedModels)
        {
            m_resourceAllocator.destroyBuffer(model.vertexBuffer);
            m_resourceAllocator.destroyBuffer(model.indexBuffer);
        }
        m_uploadedModels.clear();

        for (auto& [key, texture] : m_uploadedTextures)
        {
            m_resourceAllocator.destroyTexture(texture.texture);
            m_descriptorSystem.releaseResourceDescriptors(texture.srv);
        }
        m_uploadedTextures.clear();

        for (auto& [key, sampler] : m_uploadedSamplers)
        {
            m_descriptorSystem.releaseSamplers(sampler.descriptor);
        }
        m_uploadedSamplers.clear();

        m_gpuScene.shutdown();

    }

    void DX12Backend::ensurePathTraceResources()
    {
        const uint32_t width = m_swapchain.width();
        const uint32_t height = m_swapchain.height();
        if (width == 0 || height == 0)
        {
            return;
        }

        if (m_pathTraceResources.accumulation &&
            m_pathTraceResources.tonemap &&
            m_pathTraceResources.width == width &&
            m_pathTraceResources.height == height)
        {
            return;
        }

        destroyPathTraceResources();

        m_pathTraceResources.width = width;
        m_pathTraceResources.height = height;
        m_pathTraceResources.accumulatedSampleCount = 0;
        m_pathTraceResources.resetAccumulation = true;

        m_pathTraceResources.accumulation =
            m_resourceAllocator.createTexture({
                .width = width,
                .height = height,
                .format = TextureFormat::RGBA32_Float,
                .usage =
                    TextureUsageFlags::Storage |
                    TextureUsageFlags::Sampled,
                .memoryUsage = ResourceMemoryUsage::GpuOnly,
                .debugName = "DX12 path trace accumulation"
            });
        m_pathTraceResources.accumulationState =
            m_pathTraceResources.accumulation.initialState;

        m_pathTraceResources.tonemap =
            m_resourceAllocator.createTexture({
                .width = width,
                .height = height,
                .format = swapchainTextureFormat(),
                .usage =
                    TextureUsageFlags::Storage |
                    TextureUsageFlags::TransferSrc,
                .memoryUsage = ResourceMemoryUsage::GpuOnly,
                .debugName = "DX12 path trace tonemap"
            });
        m_pathTraceResources.tonemapState =
            m_pathTraceResources.tonemap.initialState;

        const uint32_t frameCount =
            static_cast<uint32_t>(std::max<size_t>(1, m_frameExecutor.framesInFlight()));
        m_pathTraceResources.pathTraceConstants.resize(frameCount);
        m_pathTraceResources.tonemapConstants.resize(frameCount);
        for (uint32_t i = 0; i < frameCount; ++i)
        {
            m_pathTraceResources.pathTraceConstants[i] =
                m_resourceAllocator.createBuffer({
                    .size = alignConstantBufferSize(sizeof(PathTraceConstants)),
                    .usage = BufferUsageFlags::Constant,
                    .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                    .mappedAtCreation = true,
                    .debugName = "DX12 path trace constants"
                });

            m_pathTraceResources.tonemapConstants[i] =
                m_resourceAllocator.createBuffer({
                    .size = alignConstantBufferSize(sizeof(TonemapConstants)),
                    .usage = BufferUsageFlags::Constant,
                    .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                    .mappedAtCreation = true,
                    .debugName = "DX12 path trace tonemap constants"
                });
        }

        updatePathTraceDescriptors();
    }

    void DX12Backend::ensurePathTraceSceneResources(
        const FrameContext& ctx,
        const SceneRenderView& scene)
    {
        if (!ctx.services || !ctx.services->assetManager)
        {
            if (!m_pathTraceResources.sceneSrvs.valid())
            {
                uploadPathTraceScene({});
            }
            return;
        }

        AssetManager& assets = *ctx.services->assetManager;
        const bool hasPendingModels =
            std::ranges::any_of(
                scene.models,
                [&assets](const SceneModelRenderItem& item)
                {
                    return !assets.model(item.model);
                });
        const bool retryThisFrame =
            (hasPendingModels ||
                m_pathTraceResources.sceneHadPendingModels ||
                (m_pathTraceResources.sceneTriangleCount == 0u &&
                    !scene.models.empty())) &&
            (m_pathTraceResources.lastSceneBuildFrame == UINT64_MAX ||
                ctx.frameIndex - m_pathTraceResources.lastSceneBuildFrame >=
                    30u);
        if (m_pathTraceResources.sceneVersion == scene.sceneVersion &&
            !retryThisFrame &&
            m_pathTraceResources.sceneSrvs.valid())
        {
            return;
        }

        m_pathTraceResources.lastSceneBuildFrame = ctx.frameIndex;
        const PathTraceMaterialTextureResolver materialResolver =
                [this, &assets](
                    AssetHandle modelHandle,
                    const MaterialTextureSlot& slot)
                {
                    PathTraceMaterialTextureIndices indices{};
                    indices.samplerIndex = requestSampler(nullptr);

                    const ModelAsset* model = assets.model(modelHandle);
                    if (!model ||
                        slot.textureIndex < 0 ||
                        static_cast<size_t>(slot.textureIndex) >=
                            model->textures.size())
                    {
                        return indices;
                    }

                    const TextureAsset& texture =
                        model->textures[static_cast<size_t>(slot.textureIndex)];
                    if (texture.samplerIndex >= 0 &&
                        static_cast<size_t>(texture.samplerIndex) <
                            model->samplers.size())
                    {
                        indices.samplerIndex =
                            requestSampler(
                                &model->samplers[
                                    static_cast<size_t>(texture.samplerIndex)]);
                    }

                    if (texture.imageIndex < 0 ||
                        static_cast<size_t>(texture.imageIndex) >=
                            model->images.size())
                    {
                        return indices;
                    }

                    indices.textureIndex =
                        requestTexture(
                            modelHandle,
                            static_cast<uint32_t>(texture.imageIndex),
                            model->images[static_cast<size_t>(texture.imageIndex)],
                            slot.transfer);
                    return indices;
                };
        PathTraceSceneData fallbackScene;
        const PathTraceSceneData* sharedScene = nullptr;
        if (m_rayTracingSceneService)
        {
            if (retryThisFrame)
                m_rayTracingSceneService->invalidate();
            (void)m_rayTracingSceneService->update(
                scene, assets, materialResolver);
            sharedScene = &m_rayTracingSceneService->sceneData();
        }
        else
        {
            fallbackScene = buildPathTraceSceneData(
                scene, assets, materialResolver);
            sharedScene = &fallbackScene;
        }
        if (sharedScene->triangles.empty() &&
            m_pathTraceResources.sceneSrvs.valid())
        {
            return;
        }
        uploadPathTraceScene(*sharedScene);

        m_pathTraceResources.sceneVersion = scene.sceneVersion;
        m_pathTraceResources.sceneHadPendingModels = hasPendingModels;
        m_pathTraceResources.accumulatedSampleCount = 0;
        m_pathTraceResources.resetAccumulation = true;
    }

    void DX12Backend::uploadPathTraceScene(
        const PathTraceSceneData& sceneData)
    {
        retirePathTraceSceneResources();

        std::vector<PathTraceTriangle> uploadTriangles = sceneData.triangles;
        uploadTriangles.insert(uploadTriangles.end(),
            sceneData.emissiveTriangles.begin(),
            sceneData.emissiveTriangles.end());

        m_pathTraceResources.sceneVertexCount =
            static_cast<uint32_t>(sceneData.vertices.size());
        m_pathTraceResources.sceneMaterialCount =
            static_cast<uint32_t>(sceneData.materials.size());
        m_pathTraceResources.sceneTriangleCount =
            static_cast<uint32_t>(uploadTriangles.size());
        m_pathTraceResources.sceneBvhNodeCount =
            static_cast<uint32_t>(sceneData.bvhNodes.size());
        m_pathTraceResources.firstEmissiveTriangleIndex =
            sceneData.firstEmissiveTriangleIndex;
        m_pathTraceResources.emissiveTriangleCount =
            static_cast<uint32_t>(sceneData.emissiveTriangles.size());

        spdlog::info(
            "[DX12] Path trace scene upload: vertices={} materials={} triangles={} bvhNodes={} firstEmissiveTriangle={}",
            m_pathTraceResources.sceneVertexCount,
            m_pathTraceResources.sceneMaterialCount,
            m_pathTraceResources.sceneTriangleCount,
            m_pathTraceResources.sceneBvhNodeCount,
            m_pathTraceResources.firstEmissiveTriangleIndex);

        struct PendingSceneUpload
        {
            DX12Buffer* destination = nullptr;
            DX12Buffer staging;
            uint64_t byteSize = 0;
        };

        std::vector<PendingSceneUpload> pendingUploads;
        pendingUploads.reserve(4);

        auto createBuffer =
            [&](uint64_t elementSize,
                uint32_t elementCount,
                const void* data,
                const char* debugName)
            {
                const uint64_t byteSize =
                    elementSize * std::max<uint32_t>(1u, elementCount);

                DX12Buffer buffer =
                    m_resourceAllocator.createBuffer({
                        .size = byteSize,
                        .usage = BufferUsageFlags::TransferDst,
                        .memoryUsage = ResourceMemoryUsage::GpuOnly,
                        .mappedAtCreation = false,
                        .debugName = debugName
                    });

                if (elementCount != 0u && data)
                {
                    DX12Buffer staging =
                        m_resourceAllocator.createBuffer({
                            .size = byteSize,
                            .usage = BufferUsageFlags::TransferSrc,
                            .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                            .mappedAtCreation = true,
                            .debugName = "DX12 path trace scene staging"
                        });

                    std::memcpy(
                        staging.mapped,
                        data,
                        static_cast<size_t>(elementSize * elementCount));

                    pendingUploads.push_back({
                        .destination = nullptr,
                        .staging = std::move(staging),
                        .byteSize = elementSize * elementCount
                    });
                }

                return buffer;
            };

        m_pathTraceResources.sceneVertices =
            createBuffer(
                sizeof(PathTraceVertex),
                m_pathTraceResources.sceneVertexCount,
                sceneData.vertices.data(),
                "DX12 path trace scene vertices");
        m_pathTraceResources.sceneMaterials =
            createBuffer(
                sizeof(PathTraceMaterial),
                m_pathTraceResources.sceneMaterialCount,
                sceneData.materials.data(),
                "DX12 path trace scene materials");
        m_pathTraceResources.sceneTriangles =
            createBuffer(
                sizeof(PathTraceTriangle),
                m_pathTraceResources.sceneTriangleCount,
                uploadTriangles.data(),
                "DX12 path trace scene triangles");
        m_pathTraceResources.sceneBvhNodes =
            createBuffer(
                sizeof(PathTraceBVHNode),
                m_pathTraceResources.sceneBvhNodeCount,
                sceneData.bvhNodes.data(),
                "DX12 path trace scene BVH nodes");

        uint32_t uploadIndex = 0;
        auto bindPendingUpload =
            [&](DX12Buffer& destination, uint32_t elementCount)
            {
                if (elementCount == 0u)
                {
                    return;
                }

                pendingUploads[uploadIndex++].destination = &destination;
            };

        bindPendingUpload(
            m_pathTraceResources.sceneVertices,
            m_pathTraceResources.sceneVertexCount);
        bindPendingUpload(
            m_pathTraceResources.sceneMaterials,
            m_pathTraceResources.sceneMaterialCount);
        bindPendingUpload(
            m_pathTraceResources.sceneTriangles,
            m_pathTraceResources.sceneTriangleCount);
        bindPendingUpload(
            m_pathTraceResources.sceneBvhNodes,
            m_pathTraceResources.sceneBvhNodeCount);

        if (!pendingUploads.empty())
        {
            m_uploadScheduler.record(
                [&](ID3D12GraphicsCommandList4* cmd)
                {
                    for (const PendingSceneUpload& upload : pendingUploads)
                    {
                        transitionResource(
                            cmd,
                            upload.destination->resource.Get(),
                            D3D12_RESOURCE_STATE_COMMON,
                            D3D12_RESOURCE_STATE_COPY_DEST);
                        cmd->CopyBufferRegion(
                            upload.destination->resource.Get(),
                            0,
                            upload.staging.resource.Get(),
                            0,
                            upload.byteSize);
                        transitionResource(
                            cmd,
                            upload.destination->resource.Get(),
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                    }
                });

            for (PendingSceneUpload& upload : pendingUploads)
            {
                m_uploadScheduler.retire(std::move(upload.staging));
            }
        }

        m_pathTraceResources.sceneSrvs =
            m_descriptorSystem.allocateResourceDescriptors(5);

        auto writeSrv =
            [&](const DX12Buffer& buffer,
                uint32_t elementSize,
                uint32_t elementCount,
                uint32_t descriptorIndex)
            {
                D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
                srv.Shader4ComponentMapping =
                    D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                srv.Format = DXGI_FORMAT_UNKNOWN;
                srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                srv.Buffer.NumElements =
                    std::max<UINT>(1u, elementCount);
                srv.Buffer.StructureByteStride = elementSize;

                D3D12_CPU_DESCRIPTOR_HANDLE handle =
                    m_pathTraceResources.sceneSrvs.cpuStart;
                handle.ptr +=
                    static_cast<SIZE_T>(descriptorIndex) *
                    m_pathTraceResources.sceneSrvs.descriptorSize;

                m_device.device()->CreateShaderResourceView(
                    buffer.resource.Get(),
                    &srv,
                    handle);
            };

        writeSrv(
            m_pathTraceResources.sceneVertices,
            sizeof(PathTraceVertex),
            m_pathTraceResources.sceneVertexCount,
            0);
        writeSrv(
            m_pathTraceResources.sceneMaterials,
            sizeof(PathTraceMaterial),
            m_pathTraceResources.sceneMaterialCount,
            1);
        writeSrv(
            m_pathTraceResources.sceneTriangles,
            sizeof(PathTraceTriangle),
            m_pathTraceResources.sceneTriangleCount,
            2);
        writeSrv(
            m_pathTraceResources.sceneBvhNodes,
            sizeof(PathTraceBVHNode),
            m_pathTraceResources.sceneBvhNodeCount,
            3);
    }

    void DX12Backend::destroyPathTraceSceneResources()
    {
        m_resourceAllocator.destroyBuffer(
            m_pathTraceResources.sceneVertices);
        m_resourceAllocator.destroyBuffer(
            m_pathTraceResources.sceneMaterials);
        m_resourceAllocator.destroyBuffer(
            m_pathTraceResources.sceneTriangles);
        m_resourceAllocator.destroyBuffer(
            m_pathTraceResources.sceneBvhNodes);

        m_descriptorSystem.releaseResourceDescriptors(
            m_pathTraceResources.sceneSrvs);
        m_pathTraceResources.sceneSrvs = {};
        m_pathTraceResources.sceneVertexCount = 0;
        m_pathTraceResources.sceneMaterialCount = 0;
        m_pathTraceResources.sceneTriangleCount = 0;
        m_pathTraceResources.sceneBvhNodeCount = 0;
        m_pathTraceResources.firstEmissiveTriangleIndex = UINT32_MAX;
    }

    void DX12Backend::retirePathTraceSceneResources()
    {
        m_retirementQueue.retire(
            std::move(m_pathTraceResources.sceneVertices));
        m_retirementQueue.retire(
            std::move(m_pathTraceResources.sceneMaterials));
        m_retirementQueue.retire(
            std::move(m_pathTraceResources.sceneTriangles));
        m_retirementQueue.retire(
            std::move(m_pathTraceResources.sceneBvhNodes));
        m_retirementQueue.retire(m_pathTraceResources.sceneSrvs);
        m_pathTraceResources.sceneSrvs = {};
    }

    void DX12Backend::destroyPathTraceResources()
    {
        destroyPathTraceSceneResources();

        m_resourceAllocator.destroyTexture(
            m_pathTraceResources.accumulation);
        m_resourceAllocator.destroyTexture(
            m_pathTraceResources.tonemap);
        for (DX12Buffer& buffer : m_pathTraceResources.pathTraceConstants)
        {
            m_resourceAllocator.destroyBuffer(buffer);
        }
        for (DX12Buffer& buffer : m_pathTraceResources.tonemapConstants)
        {
            m_resourceAllocator.destroyBuffer(buffer);
        }

        m_descriptorSystem.releaseResourceDescriptors(
            m_pathTraceResources.accumulationUav);
        m_descriptorSystem.releaseResourceDescriptors(
            m_pathTraceResources.accumulationSrv);
        m_descriptorSystem.releaseResourceDescriptors(
            m_pathTraceResources.tonemapUav);

        m_pathTraceResources = {};
    }

    void DX12Backend::updatePathTraceDescriptors()
    {
        if (!m_pathTraceResources.accumulation ||
            !m_pathTraceResources.tonemap)
        {
            return;
        }

        m_descriptorSystem.releaseResourceDescriptors(
            m_pathTraceResources.accumulationUav);
        m_descriptorSystem.releaseResourceDescriptors(
            m_pathTraceResources.accumulationSrv);
        m_descriptorSystem.releaseResourceDescriptors(
            m_pathTraceResources.tonemapUav);

        m_pathTraceResources.accumulationUav =
            m_descriptorSystem.allocateResourceDescriptors(1);
        m_pathTraceResources.accumulationSrv =
            m_descriptorSystem.allocateResourceDescriptors(1);
        m_pathTraceResources.tonemapUav =
            m_descriptorSystem.allocateResourceDescriptors(1);

        D3D12_UNORDERED_ACCESS_VIEW_DESC accumulationUav{};
        accumulationUav.Format =
            m_pathTraceResources.accumulation.desc.Format;
        accumulationUav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;

        m_device.device()->CreateUnorderedAccessView(
            m_pathTraceResources.accumulation.resource.Get(),
            nullptr,
            &accumulationUav,
            m_pathTraceResources.accumulationUav.cpuStart);

        D3D12_SHADER_RESOURCE_VIEW_DESC accumulationSrv{};
        accumulationSrv.Shader4ComponentMapping =
            D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        accumulationSrv.Format =
            m_pathTraceResources.accumulation.desc.Format;
        accumulationSrv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        accumulationSrv.Texture2D.MipLevels = 1;

        m_device.device()->CreateShaderResourceView(
            m_pathTraceResources.accumulation.resource.Get(),
            &accumulationSrv,
            m_pathTraceResources.accumulationSrv.cpuStart);

        D3D12_UNORDERED_ACCESS_VIEW_DESC tonemapUav{};
        tonemapUav.Format = m_pathTraceResources.tonemap.desc.Format;
        tonemapUav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;

        m_device.device()->CreateUnorderedAccessView(
            m_pathTraceResources.tonemap.resource.Get(),
            nullptr,
            &tonemapUav,
            m_pathTraceResources.tonemapUav.cpuStart);
    }

    void DX12Backend::ensureComputeTestBuffer()
    {
        if (m_computeTestBuffer)
        {
            return;
        }

        m_computeTestBuffer =
            m_resourceAllocator.createBuffer({
                .size = 64 * 64 * sizeof(uint32_t),
                .usage = BufferUsageFlags::Storage,
                .memoryUsage = ResourceMemoryUsage::GpuOnly,
                .debugName = "DX12 compute binding test buffer"
            });
        m_computeTestBufferState = m_computeTestBuffer.initialState;
    }

    void DX12Backend::ensureClusteredForwardResources()
    {
        // Recording jobs for other nodes (e.g. GpuFrustumCull) may call this
        // redundantly from worker threads while a resize is in flight on the
        // main thread; the check-then-destroy-then-recreate sequence below is
        // not atomic on its own, so serialize the whole thing.
        std::lock_guard<std::mutex> lock(m_clusteredForwardResourcesMutex);

        const uint32_t width = std::max(1u, m_swapchain.width());
        const uint32_t height = std::max(1u, m_swapchain.height());
        const uint32_t clusterCountX =
            (width + ClusteredForwardTileSizeX - 1u) /
            ClusteredForwardTileSizeX;
        const uint32_t clusterCountY =
            (height + ClusteredForwardTileSizeY - 1u) /
            ClusteredForwardTileSizeY;
        const uint32_t clusterCountZ = ClusteredForwardSliceCountZ;
        const uint32_t clusterCount =
            clusterCountX * clusterCountY * clusterCountZ;
        const uint32_t hiZMipCount =
            1u + static_cast<uint32_t>(
                std::floor(std::log2(std::max(width, height))));
        const uint32_t maxInstances = ClusteredForwardMaxGpuCullInstances;

        // The cluster bounds/light-grid/index/counter buffers AND the GPU-driven
        // cull/indirect buffers are all frame-graph owned (materialized by the
        // registry from the path declarations); this method now only derives
        // cluster dimensions. clusterCount is the "initialized" marker (0 until
        // first run) since there is no backend-owned cluster/cull buffer.
        if (m_clusteredForwardResources.clusterCount != 0 &&
            m_clusteredForwardResources.width == width &&
            m_clusteredForwardResources.height == height)
        {
            return;
        }

        m_clusteredForwardResources.width = width;
        m_clusteredForwardResources.height = height;
        m_clusteredForwardResources.clusterCountX = clusterCountX;
        m_clusteredForwardResources.clusterCountY = clusterCountY;
        m_clusteredForwardResources.clusterCountZ = clusterCountZ;
        m_clusteredForwardResources.clusterCount = clusterCount;
        m_clusteredForwardResources.hiZMipCount = hiZMipCount;

        spdlog::info(
            "[DX12Backend] Clustered forward resources: {}x{}x{} clusters, Hi-Z {}x{} mips={}, maxCullInstances={}",
            clusterCountX,
            clusterCountY,
            clusterCountZ,
            width,
            height,
            hiZMipCount,
            maxInstances);
    }

    DX12GraphResourceEntry* DX12Backend::gpuDrivenBufferEntry(
        const CompiledGraphPlan& plan,
        GraphResourceSemantic semantic)
    {
        const GraphResourceId id = findResourceBySemantic(plan, semantic);
        return id != InvalidGraphResourceId
            ? m_graphResourceRegistry.entry(id)
            : nullptr;
    }

    void DX12Backend::uploadGpuDrivenInputs(
        const CompiledGraphPlan& plan,
        const FrameContext& ctx)
    {
        (void)ctx;
        // Upload a prepared CPU span into a graph-owned mapped buffer, clamped
        // to the buffer's fixed capacity.
        auto upload =
            [&](GraphResourceSemantic semantic, const void* data, size_t bytes)
            {
                DX12GraphResourceEntry* entry =
                    gpuDrivenBufferEntry(plan, semantic);
                if (!entry || !entry->buffer.mapped)
                {
                    return;
                }
                const size_t clamped = std::min<size_t>(
                    bytes, static_cast<size_t>(entry->buffer.size));
                if (clamped > 0)
                {
                    std::memcpy(entry->buffer.mapped, data, clamped);
                }
            };

        const std::span<const GpuInstanceBounds> bounds =
            m_gpuScene.preparedInstanceBounds();
        const std::span<const GpuDrawInput> inputs =
            m_gpuScene.preparedDrawInputs();
        const std::span<const GpuVisibleLight> lights =
            m_gpuScene.preparedVisibleLights();
        upload(GraphResourceSemantic::GpuDrivenInstanceBounds,
            bounds.data(), bounds.size() * sizeof(GpuInstanceBounds));
        upload(GraphResourceSemantic::GpuDrivenDrawInputs,
            inputs.data(), inputs.size() * sizeof(GpuDrawInput));
        upload(GraphResourceSemantic::VisibleLights,
            lights.data(), lights.size() * sizeof(GpuVisibleLight));
    }

    bool DX12Backend::bindClusteredForwardCompute(
        const DX12ComputePipeline& pipeline,
        const CompiledGraphPlan& plan,
        const FrameContext& ctx,
        const SceneRenderView& scene,
        ID3D12GraphicsCommandList4* cmd)
    {
        ensureClusteredForwardResources();
        if (!prepareSceneResources(ctx, scene))
        {
            return false;
        }

        const uint32_t frameSlot =
            static_cast<uint32_t>(ctx.frameIndex % m_gpuScene.frameSlotCount());
        DX12GpuSceneFrameResources& frameResources =
            m_gpuScene.frameResources(frameSlot);

        DX12PassContext passCtx{};
        passCtx.cmd = cmd;
        passCtx.plan = &plan;
        passCtx.resources = &m_graphResourceRegistry;
        passCtx.descriptors = &m_descriptorSystem;
        recordClusteredForwardCompute(
            passCtx, pipeline, frameResources.frameConstants.gpuAddress);
        return true;
    }

    void DX12Backend::destroyClusteredForwardResources()
    {
        // Cluster bounds/light-grid/index/counter buffers are frame-graph owned
        // (freed by the registry); nothing backend-managed to release here.
        m_clusteredForwardResources = {};
    }

    bool DX12Backend::ensureEnvironmentResources(
        const FrameContext& ctx,
        const SceneRenderView& scene,
        ID3D12GraphicsCommandList4* cmd)
    {
        if (scene.environment.enabled == 0u ||
            !scene.environment.equirectTexture ||
            !ctx.services ||
            !ctx.services->assetManager)
        {
            return false;
        }

        const uint32_t cubemapSize =
            std::max(1u, scene.environment.settings.cubemapSize);
        if (m_environmentResources.source != scene.environment.equirectTexture ||
            m_environmentResources.cubemapSize != cubemapSize)
        {
            destroyEnvironmentResources();
            m_environmentResources.source = scene.environment.equirectTexture;
            m_environmentResources.cubemapSize = cubemapSize;
        }

        if (!m_environmentResources.cubemap)
        {
            m_environmentResources.cubemap =
                m_resourceAllocator.createTexture({
                    .width = m_environmentResources.cubemapSize,
                    .height = m_environmentResources.cubemapSize,
                    .depth = 1,
                    .mipLevels = 1,
                    .arrayLayers = 6,
                    .cubeCompatible = true,
                    .format = TextureFormat::RGBA32_Float,
                    .usage =
                        TextureUsageFlags::Sampled |
                        TextureUsageFlags::Storage,
                    .memoryUsage = ResourceMemoryUsage::GpuOnly,
                    .debugName = "DX12 environment cubemap"
                });
            m_environmentResources.cubemapState =
                m_environmentResources.cubemap.initialState;

            m_environmentResources.cubemapSrv =
                m_descriptorSystem.allocateResourceDescriptors(1);
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.Shader4ComponentMapping =
                D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.Format = m_environmentResources.cubemap.desc.Format;
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
            srv.TextureCube.MipLevels = 1;
            m_device.device()->CreateShaderResourceView(
                m_environmentResources.cubemap.resource.Get(),
                &srv,
                m_environmentResources.cubemapSrv.cpuStart);

            m_environmentResources.cubemapUav =
                m_descriptorSystem.allocateResourceDescriptors(1);
            D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
            uav.Format = m_environmentResources.cubemap.desc.Format;
            uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            uav.Texture2DArray.ArraySize = 6;
            m_device.device()->CreateUnorderedAccessView(
                m_environmentResources.cubemap.resource.Get(),
                nullptr,
                &uav,
                m_environmentResources.cubemapUav.cpuStart);

            m_environmentResources.sampler =
                m_descriptorSystem.allocateSamplers(1);
            D3D12_SAMPLER_DESC sampler{};
            sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
            sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            sampler.MaxLOD = D3D12_FLOAT32_MAX;
            m_device.device()->CreateSampler(
                &sampler,
                m_environmentResources.sampler.cpuStart);
        }

        if (m_environmentResources.skyboxConstants.empty())
        {
            // One instance per frame-in-flight (indexed by frameIndex): with
            // multi-frame GPU overlap now enabled, adjacent frames must write
            // distinct skybox-constant buffers, and slot reuse is bounded by the
            // same frame-slot fence as the other per-frame resources.
            const uint32_t frameCount = std::max<uint32_t>(
                1u, m_frameExecutor.framesInFlight());
            m_environmentResources.skyboxConstants.resize(frameCount);
            for (DX12Buffer& buffer :
                m_environmentResources.skyboxConstants)
            {
                buffer = m_resourceAllocator.createBuffer({
                    .size = alignConstantBufferSize(sizeof(SkyboxConstants)),
                    .usage = BufferUsageFlags::Constant,
                    .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                    .mappedAtCreation = true,
                    .debugName = "DX12 skybox constants"
                });
            }
        }

        if (m_environmentResources.converted)
        {
            return true;
        }

        const ImageAsset* image =
            ctx.services->assetManager->image(scene.environment.equirectTexture);
        if (!image || !image->valid())
        {
            return true;
        }

        requestTexture(
            scene.environment.equirectTexture,
            0,
            *image,
            TextureTransferFunction::Linear);

        (void)cmd;
        return true;
    }

    void DX12Backend::drawSkybox(
        DX12GraphicsPipeline& pipeline,
        const FrameContext& ctx,
        const SceneRenderView& scene,
        ID3D12GraphicsCommandList4* cmd)
    {
        if (!m_environmentResources.converted)
        {
            if (DX12ComputePipeline* convertPipeline =
                    environmentConvertPipeline())
            {
                (void)convertEnvironmentIfReady(
                    *convertPipeline,
                    ctx,
                    scene,
                    cmd);
            }
        }

        if (!ensureEnvironmentResources(ctx, scene, cmd) ||
            !m_environmentResources.converted ||
            m_environmentResources.skyboxConstants.empty())
        {
            return;
        }

        constexpr D3D12_RESOURCE_STATES shaderReadState =
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        if (m_environmentResources.cubemapState != shaderReadState)
        {
            transitionResource(
                cmd,
                m_environmentResources.cubemap.resource.Get(),
                m_environmentResources.cubemapState,
                shaderReadState);
            m_environmentResources.cubemapState = shaderReadState;
        }

        const uint32_t frameSlot =
            static_cast<uint32_t>(
                ctx.frameIndex %
                m_environmentResources.skyboxConstants.size());
        DX12Buffer& constantsBuffer =
            m_environmentResources.skyboxConstants[frameSlot];

        DX12SkyboxInputs inputs{};
        inputs.pipeline = &pipeline;
        inputs.constantsMapped = constantsBuffer.mapped;
        inputs.constantsAddr = constantsBuffer.gpuAddress;
        inputs.cubemapSrv = m_environmentResources.cubemapSrv.gpuStart;
        inputs.sampler = m_environmentResources.sampler.gpuStart;

        DX12PassContext passCtx{};
        passCtx.cmd = cmd;
        passCtx.scene = &scene;
        passCtx.descriptors = &m_descriptorSystem;
        passCtx.surfaceWidth = m_swapchain.width();
        passCtx.surfaceHeight = m_swapchain.height();
        recordSkybox(passCtx, inputs);
    }

    void DX12Backend::destroyEnvironmentResources()
    {
        m_resourceAllocator.destroyTexture(m_environmentResources.cubemap);
        m_resourceAllocator.destroyTexture(m_environmentResources.irradiance);
        m_resourceAllocator.destroyTexture(m_environmentResources.prefiltered);
        m_resourceAllocator.destroyTexture(m_environmentResources.brdfLut);
        m_descriptorSystem.releaseResourceDescriptors(
            m_environmentResources.cubemapSrv);
        m_descriptorSystem.releaseResourceDescriptors(
            m_environmentResources.cubemapUav);
        m_descriptorSystem.releaseResourceDescriptors(
            m_environmentResources.irradianceSrv);
        m_descriptorSystem.releaseResourceDescriptors(
            m_environmentResources.irradianceUav);
        m_descriptorSystem.releaseResourceDescriptors(
            m_environmentResources.prefilteredSrv);
        for (DX12DescriptorAllocation allocation :
            m_environmentResources.prefilteredUavs)
        {
            m_descriptorSystem.releaseResourceDescriptors(allocation);
        }
        m_descriptorSystem.releaseResourceDescriptors(
            m_environmentResources.brdfLutSrv);
        m_descriptorSystem.releaseResourceDescriptors(
            m_environmentResources.brdfLutUav);
        m_descriptorSystem.releaseSamplers(m_environmentResources.sampler);
        for (DX12Buffer& buffer :
            m_environmentResources.skyboxConstants)
        {
            m_resourceAllocator.destroyBuffer(buffer);
        }
        m_environmentResources = {};
    }

    void DX12Backend::ensureDepthTarget()
    {
        const uint32_t width = m_swapchain.width();
        const uint32_t height = m_swapchain.height();
        if (m_depthTexture &&
            m_depthWidth == width &&
            m_depthHeight == height)
        {
            return;
        }

        destroyDepthTarget();

        m_depthTexture =
            m_resourceAllocator.createTexture({
                .width = width,
                .height = height,
                .format = TextureFormat::D32_Float,
                .usage =
                    TextureUsageFlags::DepthAttachment |
                    TextureUsageFlags::Sampled,
                .memoryUsage = ResourceMemoryUsage::GpuOnly,
                .debugName = "DX12 forward depth"
            });

        m_depthDsv = m_descriptorSystem.allocateDSV(1);
        D3D12_DEPTH_STENCIL_VIEW_DESC dsv{};
        dsv.Format = DXGI_FORMAT_D32_FLOAT;
        dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;

        m_device.device()->CreateDepthStencilView(
            m_depthTexture.resource.Get(),
            &dsv,
            m_depthDsv.cpuStart);

        m_depthSrv = m_descriptorSystem.allocateResourceDescriptors(1);
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = DXGI_FORMAT_R32_FLOAT;
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D.MipLevels = 1;
        m_device.device()->CreateShaderResourceView(
            m_depthTexture.resource.Get(),
            &srv,
            m_depthSrv.cpuStart);

        m_depthState = m_depthTexture.initialState;
        m_depthWidth = width;
        m_depthHeight = height;
    }

    void DX12Backend::destroyDepthTarget()
    {
        m_resourceAllocator.destroyTexture(m_depthTexture);
        m_descriptorSystem.releaseDSV(m_depthDsv);
        m_descriptorSystem.releaseResourceDescriptors(m_depthSrv);
        m_depthDsv = {};
        m_depthSrv = {};
        m_depthState = D3D12_RESOURCE_STATE_COMMON;
        m_depthWidth = 0;
        m_depthHeight = 0;
    }

    std::vector<IBLBakeResult> DX12Backend::executeIBLBakeRequests(
        std::span<const IBLBakeRequest> requests,
        const FrameContext& ctx)
    {
        std::vector<IBLBakeResult> results;
        results.reserve(requests.size());

        for (const IBLBakeRequest& request : requests)
        {
            IBLBakeResult result{};
            result.requestId = request.requestId;
            result.handle = request.handle;

            if (!ctx.services || !ctx.services->assetManager)
            {
                result.error = "SourceNotReady";
                results.push_back(std::move(result));
                continue;
            }

            const ImageAsset* image =
                ctx.services->assetManager->image(
                    request.desc.sourceEnvironment);
            if (!image || !image->valid())
            {
                result.error = "SourceNotReady";
                results.push_back(std::move(result));
                continue;
            }

            auto computePipeline = [&](const char* name) -> DX12ComputePipeline*
                {
                    const PipelineId pipelineId = makePipelineId(name);
                    const ComputePipelineHandle handle =
                        m_pipelineManager.resolveComputePipeline(
                            *m_pipelineLibrary, pipelineId);
                    return m_pipelineManager.computePipeline(handle);
                };

            DX12ComputePipeline* convertPipeline =
                computePipeline("equirect_to_cubemap");
            DX12ComputePipeline* irradiancePipeline =
                computePipeline("ibl_irradiance");
            DX12ComputePipeline* prefilterPipeline =
                computePipeline("ibl_prefilter");
            DX12ComputePipeline* brdfPipeline =
                computePipeline("ibl_brdf_lut");

            if (!convertPipeline ||
                !irradiancePipeline ||
                !prefilterPipeline ||
                !brdfPipeline)
            {
                result.error = "Missing DX12 IBL compute pipeline";
                results.push_back(std::move(result));
                continue;
            }

            try
            {
                SceneRenderView bakeScene{};
                bakeScene.environment.equirectTexture =
                    request.desc.sourceEnvironment;
                bakeScene.environment.settings.cubemapSize =
                    std::max(1u, request.desc.environmentSize);
                bakeScene.environment.settings.intensity = 1.0f;
                bakeScene.environment.version = request.requestId;
                bakeScene.environment.enabled = 1u;

                m_environmentResources.irradianceSize =
                    std::max(1u, request.desc.irradianceSize);
                m_environmentResources.prefilterSize =
                    std::max(1u, request.desc.prefilterSize);
                m_environmentResources.prefilterMipCount =
                    mipCountForSize(m_environmentResources.prefilterSize);
                m_environmentResources.brdfLutSize =
                    std::max(1u, request.desc.brdfLutSize);

                Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
                throwIfFailed(
                    m_device.device()->CreateCommandAllocator(
                        D3D12_COMMAND_LIST_TYPE_DIRECT,
                        IID_PPV_ARGS(&allocator)),
                    "Failed to create DX12 IBL bake allocator.");

                Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> cmd;
                throwIfFailed(
                    m_device.device()->CreateCommandList(
                        0,
                        D3D12_COMMAND_LIST_TYPE_DIRECT,
                        allocator.Get(),
                        nullptr,
                        IID_PPV_ARGS(&cmd)),
                    "Failed to create DX12 IBL bake command list.");

                // IBL baking is a synchronous result-producing API. Record its
                // source upload into the same direct list so the conversion
                // dispatch is ordered without a separate queue submit or wait.
                (void)requestTexture(
                    request.desc.sourceEnvironment,
                    0,
                    *image,
                    TextureTransferFunction::Linear,
                    cmd.Get());

                (void)ensureEnvironmentResources(
                    ctx,
                    bakeScene,
                    cmd.Get());
                (void)convertEnvironmentIfReady(
                    *convertPipeline,
                    ctx,
                    bakeScene,
                    cmd.Get());

                auto createCubeSrv = [&](DX12Texture& texture)
                    {
                        DX12DescriptorAllocation allocation =
                            m_descriptorSystem.allocateResourceDescriptors(1);
                        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
                        srv.Shader4ComponentMapping =
                            D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                        srv.Format = texture.desc.Format;
                        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
                        srv.TextureCube.MipLevels = texture.desc.MipLevels;
                        m_device.device()->CreateShaderResourceView(
                            texture.resource.Get(),
                            &srv,
                            allocation.cpuStart);
                        return allocation;
                    };

                auto createCubeUav = [&](DX12Texture& texture, uint32_t mip)
                    {
                        DX12DescriptorAllocation allocation =
                            m_descriptorSystem.allocateResourceDescriptors(1);
                        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
                        uav.Format = texture.desc.Format;
                        uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                        uav.Texture2DArray.MipSlice = mip;
                        uav.Texture2DArray.ArraySize = 6;
                        m_device.device()->CreateUnorderedAccessView(
                            texture.resource.Get(),
                            nullptr,
                            &uav,
                            allocation.cpuStart);
                        return allocation;
                    };

                if (!m_environmentResources.irradiance)
                {
                    m_environmentResources.irradiance =
                        m_resourceAllocator.createTexture({
                            .width = m_environmentResources.irradianceSize,
                            .height = m_environmentResources.irradianceSize,
                            .depth = 1,
                            .mipLevels = 1,
                            .arrayLayers = 6,
                            .cubeCompatible = true,
                            .format = request.desc.format,
                            .usage =
                                TextureUsageFlags::Sampled |
                                TextureUsageFlags::Storage,
                            .memoryUsage = ResourceMemoryUsage::GpuOnly,
                            .debugName = "DX12 IBL irradiance cubemap"
                            });
                    m_environmentResources.irradianceState =
                        m_environmentResources.irradiance.initialState;
                    m_environmentResources.irradianceSrv =
                        createCubeSrv(m_environmentResources.irradiance);
                    m_environmentResources.irradianceUav =
                        createCubeUav(m_environmentResources.irradiance, 0);
                }

                if (!m_environmentResources.prefiltered)
                {
                    m_environmentResources.prefiltered =
                        m_resourceAllocator.createTexture({
                            .width = m_environmentResources.prefilterSize,
                            .height = m_environmentResources.prefilterSize,
                            .depth = 1,
                            .mipLevels = m_environmentResources.prefilterMipCount,
                            .arrayLayers = 6,
                            .cubeCompatible = true,
                            .format = request.desc.format,
                            .usage =
                                TextureUsageFlags::Sampled |
                                TextureUsageFlags::Storage,
                            .memoryUsage = ResourceMemoryUsage::GpuOnly,
                            .debugName = "DX12 IBL prefiltered cubemap"
                            });
                    m_environmentResources.prefilteredState =
                        m_environmentResources.prefiltered.initialState;
                    m_environmentResources.prefilteredSrv =
                        createCubeSrv(m_environmentResources.prefiltered);
                    m_environmentResources.prefilteredUavs.clear();
                    for (uint32_t mip = 0;
                        mip < m_environmentResources.prefilterMipCount;
                        ++mip)
                    {
                        m_environmentResources.prefilteredUavs.push_back(
                            createCubeUav(
                                m_environmentResources.prefiltered,
                                mip));
                    }
                }

                if (!m_environmentResources.brdfLut)
                {
                    m_environmentResources.brdfLut =
                        m_resourceAllocator.createTexture({
                            .width = m_environmentResources.brdfLutSize,
                            .height = m_environmentResources.brdfLutSize,
                            .depth = 1,
                            .mipLevels = 1,
                            .arrayLayers = 1,
                            .format = request.desc.format,
                            .usage =
                                TextureUsageFlags::Sampled |
                                TextureUsageFlags::Storage,
                            .memoryUsage = ResourceMemoryUsage::GpuOnly,
                            .debugName = "DX12 IBL BRDF LUT"
                            });
                    m_environmentResources.brdfLutState =
                        m_environmentResources.brdfLut.initialState;
                    m_environmentResources.brdfLutSrv =
                        m_descriptorSystem.allocateResourceDescriptors(1);
                    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
                    srv.Shader4ComponentMapping =
                        D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                    srv.Format = m_environmentResources.brdfLut.desc.Format;
                    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                    srv.Texture2D.MipLevels = 1;
                    m_device.device()->CreateShaderResourceView(
                        m_environmentResources.brdfLut.resource.Get(),
                        &srv,
                        m_environmentResources.brdfLutSrv.cpuStart);

                    m_environmentResources.brdfLutUav =
                        m_descriptorSystem.allocateResourceDescriptors(1);
                    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
                    uav.Format = m_environmentResources.brdfLut.desc.Format;
                    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
                    m_device.device()->CreateUnorderedAccessView(
                        m_environmentResources.brdfLut.resource.Get(),
                        nullptr,
                        &uav,
                        m_environmentResources.brdfLutUav.cpuStart);
                }

                auto bindCompute =
                    [&](DX12ComputePipeline& pipeline,
                        D3D12_GPU_DESCRIPTOR_HANDLE source,
                        D3D12_GPU_DESCRIPTOR_HANDLE output,
                        uint32_t groupsX,
                        uint32_t groupsY,
                        uint32_t groupsZ)
                    {
                        ID3D12DescriptorHeap* heaps[] =
                        {
                            m_descriptorSystem.shaderResourceHeap(),
                            m_descriptorSystem.samplerHeap()
                        };
                        cmd->SetDescriptorHeaps(
                            static_cast<UINT>(std::size(heaps)),
                            heaps);
                        cmd->SetComputeRootSignature(
                            pipeline.rootSignature.Get());
                        cmd->SetPipelineState(pipeline.pipelineState.Get());
                        cmd->SetComputeRootDescriptorTable(0, source);
                        cmd->SetComputeRootDescriptorTable(1, output);
                        cmd->SetComputeRootDescriptorTable(
                            2,
                            m_environmentResources.sampler.gpuStart);
                        cmd->Dispatch(groupsX, groupsY, groupsZ);
                    };

                transitionResource(
                    cmd.Get(),
                    m_environmentResources.irradiance.resource.Get(),
                    m_environmentResources.irradianceState,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                m_environmentResources.irradianceState =
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                bindCompute(
                    *irradiancePipeline,
                    m_environmentResources.cubemapSrv.gpuStart,
                    m_environmentResources.irradianceUav.gpuStart,
                    (m_environmentResources.irradianceSize + 7u) / 8u,
                    (m_environmentResources.irradianceSize + 7u) / 8u,
                    6u);

                D3D12_RESOURCE_BARRIER uavBarrier{};
                uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                uavBarrier.UAV.pResource =
                    m_environmentResources.irradiance.resource.Get();
                cmd->ResourceBarrier(1, &uavBarrier);
                transitionResource(
                    cmd.Get(),
                    m_environmentResources.irradiance.resource.Get(),
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                m_environmentResources.irradianceState =
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

                transitionResource(
                    cmd.Get(),
                    m_environmentResources.prefiltered.resource.Get(),
                    m_environmentResources.prefilteredState,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                m_environmentResources.prefilteredState =
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                for (uint32_t mip = 0;
                    mip < m_environmentResources.prefilterMipCount;
                    ++mip)
                {
                    const uint32_t mipSize =
                        std::max(
                            m_environmentResources.prefilterSize >> mip,
                            1u);
                    bindCompute(
                        *prefilterPipeline,
                        m_environmentResources.cubemapSrv.gpuStart,
                        m_environmentResources.prefilteredUavs[mip].gpuStart,
                        (mipSize + 7u) / 8u,
                        (mipSize + 7u) / 8u,
                        6u);
                }
                uavBarrier.UAV.pResource =
                    m_environmentResources.prefiltered.resource.Get();
                cmd->ResourceBarrier(1, &uavBarrier);
                transitionResource(
                    cmd.Get(),
                    m_environmentResources.prefiltered.resource.Get(),
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                m_environmentResources.prefilteredState =
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

                transitionResource(
                    cmd.Get(),
                    m_environmentResources.brdfLut.resource.Get(),
                    m_environmentResources.brdfLutState,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                m_environmentResources.brdfLutState =
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                bindCompute(
                    *brdfPipeline,
                    m_environmentResources.cubemapSrv.gpuStart,
                    m_environmentResources.brdfLutUav.gpuStart,
                    (m_environmentResources.brdfLutSize + 7u) / 8u,
                    (m_environmentResources.brdfLutSize + 7u) / 8u,
                    1u);
                uavBarrier.UAV.pResource =
                    m_environmentResources.brdfLut.resource.Get();
                cmd->ResourceBarrier(1, &uavBarrier);
                transitionResource(
                    cmd.Get(),
                    m_environmentResources.brdfLut.resource.Get(),
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                m_environmentResources.brdfLutState =
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                m_environmentResources.iblBaked = true;

                throwIfFailed(
                    cmd->Close(),
                    "Failed to close DX12 IBL bake command list.");

                ID3D12CommandList* commandLists[] = { cmd.Get() };
                m_device.graphicsQueue()->ExecuteCommandLists(
                    1,
                    commandLists);
                m_frameExecutor.waitForGpu();

                if (!m_environmentResources.converted)
                {
                    result.error = "SourceNotReady";
                    results.push_back(std::move(result));
                    continue;
                }

                result.success = true;
                result.resources.environmentCubemap =
                    TextureHandle{ request.handle.index };
                result.resources.irradianceCubemap =
                    TextureHandle{ request.handle.index + 1u };
                result.resources.prefilteredCubemap =
                    TextureHandle{ request.handle.index + 2u };
                result.resources.brdfLut =
                    TextureHandle{ request.handle.index + 3u };
                result.resources.environmentBindlessIndex =
                    m_environmentResources.cubemapSrv.baseIndex;
                result.resources.irradianceBindlessIndex =
                    m_environmentResources.irradianceSrv.baseIndex;
                result.resources.prefilteredBindlessIndex =
                    m_environmentResources.prefilteredSrv.baseIndex;
                result.resources.brdfLutBindlessIndex =
                    m_environmentResources.brdfLutSrv.baseIndex;
                result.resources.prefilteredMipCount =
                    m_environmentResources.prefilterMipCount;

                spdlog::info(
                    "[DX12] IBL bake request={} produced env={} irradiance={} prefilter={} mips={} brdf={}",
                    request.requestId,
                    m_environmentResources.cubemapSize,
                    m_environmentResources.irradianceSize,
                    m_environmentResources.prefilterSize,
                    m_environmentResources.prefilterMipCount,
                    m_environmentResources.brdfLutSize);
            }
            catch (const std::exception& e)
            {
                // IBL requests execute outside the normal frame validation
                // drain, so surface any API diagnostics at the failure site.
                m_device.logValidationMessages();
                result.success = false;
                result.error = e.what();
            }

            results.push_back(std::move(result));
        }

        return results;
    }

    uint32_t DX12Backend::requestTexture(
        AssetHandle modelHandle,
        uint32_t imageIndex,
        const ImageAsset& image,
        TextureTransferFunction transfer,
        ID3D12GraphicsCommandList4* immediateCommandList)
    {
        const uint64_t key = uploadedTextureKey(modelHandle, imageIndex, transfer);
        if (auto it = m_uploadedTextures.find(key);
            it != m_uploadedTextures.end())
        {
            return it->second.srv.baseIndex;
        }

        ImageAsset uploadImage = image;
        if (transfer != TextureTransferFunction::Unknown &&
            uploadImage.format == ImageFormat::RGBA8)
        {
            uploadImage.srgb = transfer == TextureTransferFunction::SRGB;
        }
        const ImageMipChain mipChain = buildImageMipChain(uploadImage);
        const uint32_t mipCount =
            static_cast<uint32_t>(mipChain.levels.size());

        UploadedTexture uploaded{};
        uploaded.texture =
            m_resourceAllocator.createTexture(
                uploadImage,
                TextureUsageFlags::Sampled | TextureUsageFlags::TransferDst,
                "DX12 bindless model texture",
                mipCount);
        uploaded.srv =
            m_descriptorSystem.allocateResourceDescriptors(1);

        std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> footprints(mipCount);
        std::vector<UINT> rowCounts(mipCount);
        std::vector<UINT64> rowSizes(mipCount);
        UINT64 uploadBytes = 0;
        m_device.device()->GetCopyableFootprints(
            &uploaded.texture.desc,
            0,
            mipCount,
            0,
            footprints.data(),
            rowCounts.data(),
            rowSizes.data(),
            &uploadBytes);

        DX12Buffer staging =
            m_resourceAllocator.createBuffer({
                .size = uploadBytes,
                .usage = BufferUsageFlags::TransferSrc,
                .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                .mappedAtCreation = true,
                .debugName = "DX12 texture upload staging"
            });

        const uint8_t* src = reinterpret_cast<const uint8_t*>(
            mipChain.pixels.data());
        uint8_t* uploadBase = reinterpret_cast<uint8_t*>(staging.mapped);
        for (uint32_t mip = 0; mip < mipCount; ++mip)
        {
            const ImageMipLevel& level = mipChain.levels[mip];
            const uint64_t tightRowBytes = level.size / level.height;
            uint8_t* dst = uploadBase + footprints[mip].Offset;
            for (uint32_t row = 0; row < level.height; ++row)
            {
                std::memcpy(
                    dst + static_cast<uint64_t>(row) *
                        footprints[mip].Footprint.RowPitch,
                    src + level.offset + static_cast<uint64_t>(row) *
                        tightRowBytes,
                    static_cast<size_t>(tightRowBytes));
            }
        }

        auto recordUpload =
            [&](ID3D12GraphicsCommandList4* cmd, bool forCopyQueue)
            {
                if (!forCopyQueue)
                {
                    transitionResource(
                        cmd,
                        uploaded.texture.resource.Get(),
                        uploaded.texture.initialState,
                        D3D12_RESOURCE_STATE_COPY_DEST);
                }

                for (uint32_t mip = 0; mip < mipCount; ++mip)
                {
                    D3D12_TEXTURE_COPY_LOCATION dstLocation{};
                    dstLocation.pResource = uploaded.texture.resource.Get();
                    dstLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                    dstLocation.SubresourceIndex = mip;

                    D3D12_TEXTURE_COPY_LOCATION srcLocation{};
                    srcLocation.pResource = staging.resource.Get();
                    srcLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                    srcLocation.PlacedFootprint = footprints[mip];

                    cmd->CopyTextureRegion(
                        &dstLocation, 0, 0, 0, &srcLocation, nullptr);
                }
                if (!forCopyQueue)
                {
                    transitionResource(
                        cmd,
                        uploaded.texture.resource.Get(),
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                }
            };
        if (immediateCommandList)
        {
            recordUpload(immediateCommandList, false);
            uploaded.state =
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            m_retirementQueue.retire(std::move(staging));
        }
        else
        {
            m_uploadScheduler.record(
                [&](ID3D12GraphicsCommandList4* cmd)
                {
                    recordUpload(cmd, false);
                });
            uploaded.state =
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            m_uploadScheduler.retire(std::move(staging));
        }

        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Shader4ComponentMapping =
            D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Format = uploaded.texture.desc.Format;
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Texture2D.MipLevels = mipCount;

        m_device.device()->CreateShaderResourceView(
            uploaded.texture.resource.Get(),
            &srv,
            uploaded.srv.cpuStart);

        const uint32_t descriptorIndex = uploaded.srv.baseIndex;
        m_uploadedTextures.emplace(key, std::move(uploaded));
        return descriptorIndex;
    }

    uint32_t DX12Backend::requestSampler(const SamplerAsset* sampler)
    {
        const uint64_t key = samplerKey(sampler);
        if (auto it = m_uploadedSamplers.find(key);
            it != m_uploadedSamplers.end())
        {
            return it->second.descriptor.baseIndex;
        }

        auto toFilter = [](TextureFilterMode minFilter, TextureFilterMode magFilter)
        {
            const bool minLinear =
                minFilter == TextureFilterMode::Linear ||
                minFilter == TextureFilterMode::LinearMipmapNearest ||
                minFilter == TextureFilterMode::LinearMipmapLinear;
            const bool magLinear = magFilter != TextureFilterMode::Nearest;
            if (minLinear && magLinear)
            {
                return D3D12_FILTER_MIN_MAG_MIP_LINEAR;
            }
            if (!minLinear && !magLinear)
            {
                return D3D12_FILTER_MIN_MAG_MIP_POINT;
            }
            return D3D12_FILTER_MIN_LINEAR_MAG_POINT_MIP_LINEAR;
        };

        auto toAddress = [](TextureWrapMode mode)
        {
            switch (mode)
            {
            case TextureWrapMode::MirroredRepeat:
                return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
            case TextureWrapMode::ClampToEdge:
                return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            case TextureWrapMode::Repeat:
                break;
            }
            return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        };

        UploadedSampler uploaded{};
        uploaded.descriptor = m_descriptorSystem.allocateSamplers(1);

        D3D12_SAMPLER_DESC desc{};
        const D3D12_FILTER baseFilter = sampler
            ? toFilter(sampler->minFilter, sampler->magFilter)
            : D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        const bool linearSampling = !sampler ||
            (sampler->minFilter != TextureFilterMode::Nearest &&
             sampler->minFilter != TextureFilterMode::NearestMipmapNearest &&
             sampler->minFilter != TextureFilterMode::NearestMipmapLinear &&
             sampler->magFilter != TextureFilterMode::Nearest);
        desc.Filter = linearSampling ? D3D12_FILTER_ANISOTROPIC : baseFilter;
        desc.AddressU = sampler ? toAddress(sampler->wrapU) : D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        desc.AddressV = sampler ? toAddress(sampler->wrapV) : D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        desc.MaxAnisotropy = linearSampling ? 8u : 1u;
        desc.MaxLOD = D3D12_FLOAT32_MAX;

        m_device.device()->CreateSampler(&desc, uploaded.descriptor.cpuStart);

        const uint32_t descriptorIndex = uploaded.descriptor.baseIndex;
        m_uploadedSamplers.emplace(key, uploaded);
        return descriptorIndex;
    }

    DX12UploadedModel* DX12Backend::requestModel(
        AssetHandle handle,
        const AssetManager& assets)
    {
        if (!handle)
        {
            return nullptr;
        }

        auto [it, inserted] =
            m_uploadedModels.try_emplace(handle);
        DX12UploadedModel& uploaded = it->second;
        if (uploaded.uploaded)
        {
            return &uploaded;
        }

        const ModelAsset* model = assets.model(handle);
        if (!model ||
            model->vertices.empty() ||
            model->indices.empty())
        {
            return nullptr;
        }

        const uint64_t vertexBytes =
            static_cast<uint64_t>(model->vertices.size()) *
            sizeof(AssetVertex);
        const uint64_t indexBytes =
            static_cast<uint64_t>(model->indices.size()) *
            sizeof(uint32_t);

        DX12Buffer vertexStaging =
            m_resourceAllocator.createBuffer({
                .size = vertexBytes,
                .usage = BufferUsageFlags::TransferSrc,
                .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                .mappedAtCreation = true,
                .debugName = "DX12 model vertex staging"
            });

        DX12Buffer indexStaging =
            m_resourceAllocator.createBuffer({
                .size = indexBytes,
                .usage = BufferUsageFlags::TransferSrc,
                .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                .mappedAtCreation = true,
                .debugName = "DX12 model index staging"
            });

        std::memcpy(
            vertexStaging.mapped,
            model->vertices.data(),
            static_cast<size_t>(vertexBytes));
        std::memcpy(
            indexStaging.mapped,
            model->indices.data(),
            static_cast<size_t>(indexBytes));

        uploaded.vertexBuffer =
            m_resourceAllocator.createBuffer({
                .size = vertexBytes,
                .usage =
                    BufferUsageFlags::Vertex |
                    BufferUsageFlags::Storage |
                    BufferUsageFlags::AccelerationStructureBuildInput |
                    BufferUsageFlags::TransferDst,
                .memoryUsage = ResourceMemoryUsage::GpuOnly,
                .mappedAtCreation = false,
                .debugName = "DX12 model vertex buffer"
            });

        uploaded.indexBuffer =
            m_resourceAllocator.createBuffer({
                .size = indexBytes,
                .usage =
                    BufferUsageFlags::Index |
                    BufferUsageFlags::Storage |
                    BufferUsageFlags::AccelerationStructureBuildInput |
                    BufferUsageFlags::TransferDst,
                .memoryUsage = ResourceMemoryUsage::GpuOnly,
                .mappedAtCreation = false,
                .debugName = "DX12 model index buffer"
            });

        m_uploadScheduler.record(
            [&](ID3D12GraphicsCommandList4* cmd)
            {
                transitionResource(
                    cmd,
                    uploaded.vertexBuffer.resource.Get(),
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_DEST);
                transitionResource(
                    cmd,
                    uploaded.indexBuffer.resource.Get(),
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_DEST);
                cmd->CopyBufferRegion(
                    uploaded.vertexBuffer.resource.Get(),
                    0,
                    vertexStaging.resource.Get(),
                    0,
                    vertexBytes);
                cmd->CopyBufferRegion(
                    uploaded.indexBuffer.resource.Get(),
                    0,
                    indexStaging.resource.Get(),
                    0,
                    indexBytes);
                transitionResource(
                    cmd,
                    uploaded.vertexBuffer.resource.Get(),
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
                transitionResource(
                    cmd,
                    uploaded.indexBuffer.resource.Get(),
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_INDEX_BUFFER);
            });
        m_uploadScheduler.retire(std::move(vertexStaging));
        m_uploadScheduler.retire(std::move(indexStaging));

        uploaded.samplerDescriptorIndices.assign(
            model->samplers.size(),
            UINT32_MAX);
        for (size_t samplerIndex = 0;
             samplerIndex < model->samplers.size();
             ++samplerIndex)
        {
            uploaded.samplerDescriptorIndices[samplerIndex] =
                requestSampler(&model->samplers[samplerIndex]);
        }
        const uint32_t defaultSampler = requestSampler(nullptr);

        auto textureDescriptor = [&](const MaterialTextureSlot& slot)
        {
            if (slot.textureIndex < 0 ||
                static_cast<size_t>(slot.textureIndex) >= model->textures.size())
            {
                return UINT32_MAX;
            }

            const TextureAsset& texture =
                model->textures[static_cast<size_t>(slot.textureIndex)];
            if (texture.imageIndex < 0 ||
                static_cast<size_t>(texture.imageIndex) >= model->images.size())
            {
                return UINT32_MAX;
            }

            return requestTexture(
                handle,
                static_cast<uint32_t>(texture.imageIndex),
                model->images[static_cast<size_t>(texture.imageIndex)],
                slot.transfer);
        };

        auto samplerDescriptor = [&](const MaterialTextureSlot& slot)
        {
            if (slot.textureIndex < 0 ||
                static_cast<size_t>(slot.textureIndex) >= model->textures.size())
            {
                return defaultSampler;
            }

            const TextureAsset& texture =
                model->textures[static_cast<size_t>(slot.textureIndex)];
            if (texture.samplerIndex < 0 ||
                static_cast<size_t>(texture.samplerIndex) >=
                    uploaded.samplerDescriptorIndices.size())
            {
                return defaultSampler;
            }

            const uint32_t descriptor =
                uploaded.samplerDescriptorIndices[
                    static_cast<size_t>(texture.samplerIndex)];
            return descriptor != UINT32_MAX ? descriptor : defaultSampler;
        };

        uploaded.materials.reserve(std::max<size_t>(1, model->materials.size()));
        for (const MaterialAsset& src : model->materials)
        {
            GpuMaterialData dst = makeGpuMaterialData(src);
            dst.baseColorTextureIndex = textureDescriptor(src.baseColorTexture);
            dst.normalTextureIndex = textureDescriptor(src.normalTexture);
            dst.metallicRoughnessTextureIndex =
                textureDescriptor(src.metallicRoughnessTexture);
            dst.occlusionTextureIndex =
                textureDescriptor(src.occlusionTexture);
            dst.emissiveTextureIndex =
                textureDescriptor(src.emissiveTexture);
            dst.baseColorSamplerIndex =
                samplerDescriptor(src.baseColorTexture);
            dst.normalSamplerIndex =
                samplerDescriptor(src.normalTexture);
            dst.metallicRoughnessSamplerIndex =
                samplerDescriptor(src.metallicRoughnessTexture);
            dst.occlusionSamplerIndex =
                samplerDescriptor(src.occlusionTexture);
            dst.emissiveSamplerIndex =
                samplerDescriptor(src.emissiveTexture);
            uploaded.materials.push_back(dst);
        }
        if (uploaded.materials.empty())
        {
            uploaded.materials.push_back(GpuMaterialData{});
        }

        std::vector<glm::mat4> meshNodeTransforms(
            model->meshes.size(),
            glm::mat4(1.0f));

        for (const NodeAsset& node : model->nodes)
        {
            if (node.meshIndex >= 0 &&
                static_cast<size_t>(node.meshIndex) < meshNodeTransforms.size())
            {
                meshNodeTransforms[static_cast<size_t>(node.meshIndex)] =
                    node.worldMatrix;
            }
        }

        for (size_t meshIndex = 0; meshIndex < model->meshes.size(); ++meshIndex)
        {
            const MeshAsset& mesh = model->meshes[meshIndex];
            for (const MeshPrimitiveAsset& primitive : mesh.primitives)
            {
                GpuMesh gpuMesh{};
                gpuMesh.materialIndex =
                    primitive.materialIndex != UINT32_MAX
                        ? primitive.materialIndex
                        : 0u;
                gpuMesh.indexCount = primitive.indexCount;
                gpuMesh.firstIndex = primitive.firstIndex;
                gpuMesh.vertexOffset = primitive.firstVertex;
                const glm::vec3 center =
                    (primitive.bounds.min + primitive.bounds.max) * 0.5f;
                const glm::vec3 extent =
                    (primitive.bounds.max - primitive.bounds.min) * 0.5f;
                gpuMesh.bounds.centerRadius =
                    glm::vec4(center, glm::length(extent));
                uploaded.meshes.push_back(gpuMesh);
                uploaded.meshTransforms.push_back(meshNodeTransforms[meshIndex]);
            }
        }

        uploaded.uploaded = true;
        return &uploaded;
    }

    bool DX12Backend::prepareGiRayQueryResources(const FrameContext& ctx)
    {
        if (!m_rayTracingSceneService ||
            !m_rayTracingSceneService->statistics().valid ||
            m_gpuScene.frameSlotCount() == 0)
            return false;

        const auto geometries = m_rayTracingSceneService->gpuGeometries();
        const auto instances = m_rayTracingSceneService->gpuInstances();
        if (geometries.empty() || instances.empty())
            return false;
        for (const auto& geometry : geometries)
            if (geometry.mapping.x >= GiMaxGeometryBufferCount)
                return false;

        const uint32_t frameSlot = static_cast<uint32_t>(
            ctx.frameIndex % m_gpuScene.frameSlotCount());
        DX12GpuSceneFrameResources& frame =
            m_gpuScene.frameResources(frameSlot);
        const auto ensureBuffer = [&](DX12Buffer& buffer, uint32_t& capacity,
            uint32_t required, uint32_t stride, const char* name)
        {
            if (required <= capacity && buffer)
                return;
            m_resourceAllocator.destroyBuffer(buffer);
            buffer = m_resourceAllocator.createBuffer({
                .size = static_cast<uint64_t>(required) * stride,
                .usage = BufferUsageFlags::None,
                .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                .mappedAtCreation = true,
                .debugName = name });
            capacity = required;
            frame.giRayQueryGeneration = UINT64_MAX;
        };
        ensureBuffer(frame.giRtGeometries, frame.giGeometryCapacity,
            static_cast<uint32_t>(geometries.size()),
            sizeof(GpuRayTracingGeometryRecord), "DX12 GI RT geometry table");
        ensureBuffer(frame.giRtInstances, frame.giInstanceCapacity,
            static_cast<uint32_t>(instances.size()),
            sizeof(GpuRayTracingInstanceRecord), "DX12 GI RT instance table");
        if (!frame.giTraceConstants)
        {
            // Constant buffers must be 256-byte aligned; the 128-byte payload is
            // written per frame in recordCompute (all GI passes write identical
            // bytes into this per-frame-slot buffer).
            frame.giTraceConstants = m_resourceAllocator.createBuffer({
                .size = 256u,
                .usage = BufferUsageFlags::Constant,
                .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                .mappedAtCreation = true,
                .debugName = "DX12 GI trace constants" });
        }
        if (!frame.giModelBufferSrvs.valid())
        {
            frame.giModelBufferSrvs =
                m_descriptorSystem.allocateResourceDescriptors(
                    GiMaxGeometryBufferCount * 2u + 1u);
            frame.giRayQueryGeneration = UINT64_MAX;
        }

        const uint64_t generation =
            m_rayTracingSceneService->statistics().generation;
        if (frame.giRayQueryGeneration == generation)
            return true;
        std::memcpy(frame.giRtGeometries.mapped, geometries.data(),
            geometries.size_bytes());
        std::memcpy(frame.giRtInstances.mapped, instances.data(),
            instances.size_bytes());

        std::array<const DX12UploadedModel*, GiMaxGeometryBufferCount> models{};
        const DX12UploadedModel* fallback = nullptr;
        for (const RayTracingGeometryRecord& geometry :
            m_rayTracingSceneService->geometries())
        {
            const auto found = m_uploadedModels.find(geometry.model);
            if (found == m_uploadedModels.end() || !found->second.uploaded)
                return false;
            models[geometry.modelBufferIndex] = &found->second;
            if (!fallback) fallback = &found->second;
        }
        if (!fallback) return false;

        const auto writeRawSrv = [&](uint32_t descriptorIndex,
            const DX12Buffer& buffer)
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            srv.Format = DXGI_FORMAT_R32_TYPELESS;
            srv.Buffer.NumElements = static_cast<UINT>(buffer.size / 4u);
            srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            D3D12_CPU_DESCRIPTOR_HANDLE handle =
                frame.giModelBufferSrvs.cpuStart;
            handle.ptr += static_cast<SIZE_T>(descriptorIndex) *
                frame.giModelBufferSrvs.descriptorSize;
            m_device.device()->CreateShaderResourceView(
                buffer.resource.Get(), &srv, handle);
        };
        for (uint32_t slot = 0; slot < GiMaxGeometryBufferCount; ++slot)
        {
            const DX12UploadedModel* model = models[slot]
                ? models[slot] : fallback;
            writeRawSrv(slot, model->vertexBuffer);
            writeRawSrv(GiMaxGeometryBufferCount + slot, model->indexBuffer);
        }
        D3D12_SHADER_RESOURCE_VIEW_DESC nullCube{};
        nullCube.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        nullCube.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
        nullCube.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        nullCube.TextureCube.MipLevels = 1;
        D3D12_CPU_DESCRIPTOR_HANDLE nullCubeHandle =
            frame.giModelBufferSrvs.cpuStart;
        nullCubeHandle.ptr += static_cast<SIZE_T>(
            GiMaxGeometryBufferCount * 2u) *
            frame.giModelBufferSrvs.descriptorSize;
        m_device.device()->CreateShaderResourceView(
            nullptr, &nullCube, nullCubeHandle);
        frame.giRayQueryGeneration = generation;
        return true;
    }

    bool DX12Backend::prepareSceneResources(
        const FrameContext& ctx,
        const SceneRenderView& scene)
    {
        if (!ctx.services || !ctx.services->assetManager)
        {
            return false;
        }
        AssetManager& assets = *ctx.services->assetManager;

        m_uploadedModels.reserve(
            m_uploadedModels.size() + scene.models.size());

        const uint32_t frameSlot =
            static_cast<uint32_t>(
                ctx.frameIndex % std::max<uint32_t>(1, m_gpuScene.frameSlotCount()));

        const bool prepared = m_gpuScene.prepare(
            ctx.frameIndex,
            scene,
            frameSlot,
            [this, &assets](AssetHandle handle) -> GpuSceneModelView
            {
                DX12UploadedModel* model = requestModel(handle, assets);
                if (!model)
                {
                    return {};
                }
                return GpuSceneModelView{
                    handle, model->meshes, model->meshTransforms,
                    model->materials };
            },
            [this, &ctx, &scene](
                uint32_t visibleLightCount,
                uint32_t instanceBoundsCount,
                uint32_t geometryBinCount) -> GpuFrameData
            {
                // Reads cluster grid dimensions this call establishes, so it
                // must run before the fields below are consumed.
                ensureClusteredForwardResources();

                GpuFrameData frameData{};
                frameData.view = scene.camera.view;
                // Swap finite near/far inputs to map near->1 and far->0. D32
                // floating-point precision then follows view-space distance
                // instead of collapsing at the far end of the scene.
                frameData.projection = glm::perspectiveRH_ZO(
                    scene.camera.verticalFovRadians,
                    scene.camera.aspectRatio,
                    scene.camera.farPlane,
                    scene.camera.nearPlane);
                frameData.viewProjection = frameData.projection * frameData.view;
                const bool occlusionHistoryReliable =
                    updateGpuOcclusionHistory(
                        m_gpuOcclusionHistory,
                        scene,
                        m_clusteredForwardResources.width,
                        m_clusteredForwardResources.height,
                        frameData.view,
                        frameData.projection,
                        frameData);
                frameData.cameraPosition = scene.camera.position;
                frameData.time = ctx.timeSinceStart;
                frameData.environmentEnabled =
                    m_environmentResources.iblBaked ? 1u : 0u;
                frameData.prefilteredMipCount =
                    m_environmentResources.prefilterMipCount;
                frameData.environmentIntensity =
                    scene.environment.settings.intensity;
                frameData.environmentExposure =
                    m_resolvedDiffuseGi != InvalidGraphResourceId
                        ? scene.environment.settings.pathTraceExposure
                        : scene.environment.settings.skyboxExposure;
                frameData.environmentTransportExposure =
                    scene.environment.settings.pathTraceExposure;

                for (const SceneLightRenderItem& light : scene.lights)
                {
                    if (light.type == LightType::Directional)
                    {
                        frameData.lightDirection = light.direction;
                        frameData.lightColor = light.color;
                        frameData.lightIntensity = light.intensity;
                        break;
                    }
                }
                for (const SceneLightRenderItem& light : scene.lights)
                {
                    if (light.type != LightType::Point ||
                        frameData.pointLightCount >= MaxGpuPointLights)
                    {
                        continue;
                    }

                    const uint32_t lightIndex = frameData.pointLightCount++;
                    frameData.pointLightPositionRange[lightIndex] =
                        glm::vec4(light.position, light.range);
                    frameData.pointLightColorIntensity[lightIndex] =
                        glm::vec4(light.color, light.intensity);
                }
                frameData.pointLightCount =
                    std::min<uint32_t>(
                        visibleLightCount, ClusteredForwardMaxVisibleLights);
                frameData.clusterDimensions = glm::uvec4(
                    m_clusteredForwardResources.clusterCountX,
                    m_clusteredForwardResources.clusterCountY,
                    m_clusteredForwardResources.clusterCountZ,
                    m_clusteredForwardResources.clusterCount);
                frameData.clusterConfig = glm::uvec4(
                    ClusteredForwardTileSizeX,
                    ClusteredForwardTileSizeY,
                    ClusteredForwardMaxLightsPerCluster,
                    m_clusteredForwardHeatmapEnabled ? 1u : 0u);
                frameData.renderExtentAndHiZ = glm::uvec4(
                    m_clusteredForwardResources.width,
                    m_clusteredForwardResources.height,
                    m_clusteredForwardResources.hiZMipCount,
                    1u);
                frameData.cullingConfig = glm::uvec4(
                    std::min<uint32_t>(
                        instanceBoundsCount, ClusteredForwardMaxGpuCullInstances),
                    // DX12 commands deliver draw identity through root
                    // constants; Vulkan uses firstInstance + metadata.
                    0u,
                    geometryBinCount,
                    occlusionHistoryReliable ? 1u : 0u);
                frameData.cameraNearFar = glm::vec4(
                    scene.camera.nearPlane, scene.camera.farPlane, 0.0f, 0.0f);
                frameData.occlusionDebugConfig = glm::uvec4(
                    static_cast<uint32_t>(m_gpuCullDebugMode),
                    m_gpuCullDebugMode != GpuCullDebugMode::Off ? 1u : 0u,
                    m_gpuOcclusionEnabled ? 1u : 0u,
                    0u);
                frameData.globalIlluminationConfig.x =
                    m_resolvedDiffuseGi != InvalidGraphResourceId ? 1u : 0u;
                frameData.globalIlluminationConfig.y = m_giDebugView;
                frameData.globalIlluminationConfig.z =
                    m_giDiagnosticIntensityBits;
                frameData.globalIlluminationConfig.w =
                    m_giDebugExposureBits;
                return frameData;
            });
        if (prepared && m_rayTracingSceneService)
        {
            m_rayTracingSceneService->updateGeometry(scene, assets);
            m_accelerationStructures.prepare(
                *m_rayTracingSceneService, m_uploadedModels);
        }
        return prepared;
    }

    void DX12Backend::initImGui(Window& window)
    {
        if (m_imguiEnabled)
        {
            return;
        }

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();

        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.IniFilename = nullptr;

        ImGui::StyleColorsDark();

        auto* glfwWindow =
            static_cast<GLFWwindow*>(window.getNativeHandle());
        if (!ImGui_ImplGlfw_InitForOther(glfwWindow, true))
        {
            throw std::runtime_error(
                "Failed to initialize ImGui GLFW backend.");
        }

        ImGui_ImplDX12_InitInfo initInfo{};
        initInfo.Device = m_device.device();
        initInfo.CommandQueue = m_device.graphicsQueue();
        initInfo.NumFramesInFlight =
            static_cast<int>(std::max<size_t>(1, m_frameExecutor.framesInFlight()));
        initInfo.RTVFormat = m_swapchain.format();
        initInfo.DSVFormat = DXGI_FORMAT_UNKNOWN;
        initInfo.SrvDescriptorHeap =
            m_descriptorSystem.shaderResourceHeap();
        initInfo.UserData = this;
        initInfo.SrvDescriptorAllocFn =
            [](ImGui_ImplDX12_InitInfo* info,
                D3D12_CPU_DESCRIPTOR_HANDLE* outCpu,
                D3D12_GPU_DESCRIPTOR_HANDLE* outGpu)
            {
                auto* self = static_cast<DX12Backend*>(info->UserData);
                DX12DescriptorAllocation allocation =
                    self->m_descriptorSystem.allocateResourceDescriptors(1);
                *outCpu = allocation.cpuStart;
                *outGpu = allocation.gpuStart;
            };
        initInfo.SrvDescriptorFreeFn =
            [](ImGui_ImplDX12_InitInfo*,
                D3D12_CPU_DESCRIPTOR_HANDLE,
                D3D12_GPU_DESCRIPTOR_HANDLE)
            {
            };

        if (!ImGui_ImplDX12_Init(&initInfo))
        {
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
            throw std::runtime_error(
                "Failed to initialize ImGui DX12 backend.");
        }

        m_imguiEnabled = true;
    }

    void DX12Backend::shutdownImGui()
    {
        if (!m_imguiEnabled)
        {
            return;
        }

        ImGui_ImplDX12_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        m_imguiIniPath.clear();

        m_imguiEnabled = false;
        m_imguiFrameActive = false;
    }


    bool DX12Backend::beginDebugGuiFrame()
    {
        if (!m_imguiEnabled || m_imguiFrameActive)
        {
            return false;
        }

        ImGui_ImplDX12_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        m_imguiFrameActive = true;
        return true;
    }

    HiZDebugImage DX12Backend::hiZDebugImage(bool previous, uint32_t mip)
    {
        const DX12GraphResourceEntry* hiZ =
            previous
                ? m_graphResourceRegistry.previousEntry(
                    m_clusteredForwardResources.hiZDebugResource)
                : m_graphResourceRegistry.entry(
                    m_clusteredForwardResources.hiZDebugResource);

        if (!hiZ || hiZ->mipLevels() == 0)
        {
            return {};
        }

        const uint32_t clamped = std::min(mip, hiZ->mipLevels() - 1u);
        if (clamped >= hiZ->mipSrvs.size())
        {
            return {};
        }

        return {
            .textureId = hiZ->mipSrvs[clamped].gpuStart.ptr,
            .width = std::max(1u, hiZ->width() >> clamped),
            .height = std::max(1u, hiZ->height() >> clamped),
            .mipLevels = hiZ->mipLevels(),
            .valid = true
        };
    }

    void DX12Backend::buildBackendDiagnostics()
    {
        const DXGI_ADAPTER_DESC3& adapter = m_adapter.desc();
        m_diagnosticAdapterName.clear();
        for (const wchar_t wide : adapter.Description)
        {
            if (wide == L'\0')
            {
                break;
            }
            // Adapter names are ASCII in practice; this avoids dragging in a
            // locale-aware conversion for a debug string.
            m_diagnosticAdapterName.push_back(
                wide < 128 ? static_cast<char>(wide) : '?');
        }

        const DX12FeatureSupport& features = m_device.features();
        m_diagnosticFeatures.clear();
        m_diagnosticFeatures.push_back({
            .name = "Debug/validation layer",
            .enabled = m_factory.validationEnabled() });
        m_diagnosticFeatures.push_back({
            .name = "Async compute queue",
            .enabled = m_device.computeQueue() != nullptr,
            .detail = "Separate D3D12 compute engine" });
        m_diagnosticFeatures.push_back({
            .name = "Bindless resources",
            .enabled = features.bindlessResources });
        m_diagnosticFeatures.push_back({
            .name = "Descriptor indexing",
            .enabled = features.descriptorIndexing });
        m_diagnosticFeatures.push_back({
            .name = "Direct heap indexing (SM6.6)",
            .enabled = features.directHeapIndexing,
            .detail = "ResourceDescriptorHeap[] without a bound table" });
        m_diagnosticFeatures.push_back({
            .name = "GPU virtual address",
            .enabled = features.gpuVirtualAddress });
        m_diagnosticFeatures.push_back({
            .name = "GPU pass timestamps",
            .enabled = m_gpuProfiler.enabled(),
            .detail = "Per-queue calibrated timestamps feeding the async "
                      "policy" });

        m_diagnosticLimits.clear();
        m_diagnosticLimits.push_back({
            .name = "Resource binding tier",
            .value = static_cast<uint64_t>(features.resourceBindingTier) });
        m_diagnosticLimits.push_back({
            .name = "Root signature version (0x_)",
            .value = static_cast<uint64_t>(features.rootSignatureVersion) });
        m_diagnosticLimits.push_back({
            .name = "Shader model (0x_)",
            .value = static_cast<uint64_t>(features.shaderModel) });
        m_diagnosticLimits.push_back({
            .name = "Dedicated video memory (MB)",
            .value = static_cast<uint64_t>(
                adapter.DedicatedVideoMemory >> 20) });
        m_diagnosticLimits.push_back({
            .name = "Frames in flight",
            .value = m_frameExecutor.framesInFlight() });
        m_diagnosticLimits.push_back({
            .name = "Recording worker slots",
            .value = m_workerSlots });
    }

    BackendDiagnosticInfo DX12Backend::backendDiagnostics() const
    {
        return {
            .backendName = "DX12",
            .adapterName = m_diagnosticAdapterName.c_str(),
            .features = std::span<const BackendFeature>(m_diagnosticFeatures),
            .limits = std::span<const BackendLimit>(m_diagnosticLimits)
        };
    }

    void DX12Backend::endDebugGuiFrame()
    {
        if (!m_imguiFrameActive)
        {
            return;
        }

        ImGui::Render();
        m_imguiFrameActive = false;
    }

    void DX12Backend::recordImGui(
        const FrameContext& ctx,
        ID3D12Resource* swapchainImage,
        std::vector<ID3D12CommandList*>& commandLists)
    {
        if (!m_imguiEnabled)
        {
            return;
        }

        ImDrawData* drawData = ImGui::GetDrawData();
        if (!drawData || drawData->TotalVtxCount == 0)
        {
            return;
        }

        auto lease =
            m_commandSystem.acquireFrameCommandList(
                static_cast<uint32_t>(ctx.frameIndex % m_frameExecutor.framesInFlight()),
                0);

        ID3D12GraphicsCommandList4* cmd =
            lease.commandList();

        DX12GraphResourceEntry* hiZDebugResource = nullptr;
        if (m_hiZDebugViewEnabled)
        {
            hiZDebugResource =
                m_hiZDebugPrevious
                    ? m_graphResourceRegistry.previousEntry(
                        m_clusteredForwardResources.hiZDebugResource)
                    : m_graphResourceRegistry.entry(
                        m_clusteredForwardResources.hiZDebugResource);
        }

        const D3D12_RESOURCE_STATES hiZOriginalState =
            hiZDebugResource
                ? hiZDebugResource->state
                : D3D12_RESOURCE_STATE_COMMON;
        if (hiZDebugResource &&
            hiZDebugResource->nativeResource() &&
            hiZOriginalState != D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)
        {
            transitionResource(
                cmd,
                hiZDebugResource->nativeResource(),
                hiZOriginalState,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        }

        transitionResource(
            cmd,
            swapchainImage,
            D3D12_RESOURCE_STATE_PRESENT,
            D3D12_RESOURCE_STATE_RENDER_TARGET);

        D3D12_CPU_DESCRIPTOR_HANDLE rtv =
            m_swapchain.currentRtv();
        cmd->OMSetRenderTargets(1, &rtv, FALSE, nullptr);

        ID3D12DescriptorHeap* heaps[] =
        {
            m_descriptorSystem.shaderResourceHeap()
        };
        cmd->SetDescriptorHeaps(1, heaps);

        ImGui_ImplDX12_RenderDrawData(drawData, cmd);

        if (hiZDebugResource &&
            hiZDebugResource->nativeResource() &&
            hiZOriginalState != D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)
        {
            transitionResource(
                cmd,
                hiZDebugResource->nativeResource(),
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                hiZOriginalState);
        }

        transitionResource(
            cmd,
            swapchainImage,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PRESENT);

        throwIfFailed(
            cmd->Close(),
            "Failed to close DX12 ImGui command list.");

        commandLists.push_back(
            static_cast<ID3D12CommandList*>(cmd));
    }



    bool DX12Backend::vsyncEnabled() const
    {
        return m_swapchain.vsyncEnabled();
    }

    void DX12Backend::setVsyncEnabled(bool enabled)
    {
        m_swapchain.setVsyncEnabled(enabled);
    }

    bool DX12Backend::clusteredForwardHeatmapEnabled() const
    {
        return m_clusteredForwardHeatmapEnabled;
    }

    void DX12Backend::setClusteredForwardHeatmapEnabled(bool enabled)
    {
        m_clusteredForwardHeatmapEnabled = enabled;
    }

    bool DX12Backend::hiZDebugViewEnabled() const
    {
        return m_hiZDebugViewEnabled;
    }

    void DX12Backend::setHiZDebugViewEnabled(bool enabled)
    {
        if (m_hiZDebugViewEnabled != enabled)
        {
            spdlog::info(
                "[DX12Backend] Hi-Z debug view {}",
                enabled ? "enabled" : "disabled");
        }
        m_hiZDebugViewEnabled = enabled;
    }

    uint32_t DX12Backend::hiZDebugMip() const
    {
        return m_hiZDebugMip;
    }

    void DX12Backend::setHiZDebugMip(uint32_t mip)
    {
        m_hiZDebugMip = mip;
    }

    bool DX12Backend::hiZDebugPrevious() const
    {
        return m_hiZDebugPrevious;
    }

    void DX12Backend::setHiZDebugPrevious(bool previous)
    {
        m_hiZDebugPrevious = previous;
    }

    bool DX12Backend::gpuOcclusionEnabled() const
    {
        return m_gpuOcclusionEnabled;
    }

    void DX12Backend::setGpuOcclusionEnabled(bool enabled)
    {
        m_gpuOcclusionEnabled = enabled;
    }

    GpuCullDebugMode DX12Backend::gpuCullDebugMode() const
    {
        return m_gpuCullDebugMode;
    }

    void DX12Backend::setGpuCullDebugMode(GpuCullDebugMode mode)
    {
        if (m_gpuCullDebugMode != mode)
        {
            m_gpuCullDiagnosticFrames = 0;
            m_gpuCullStats = {};
            m_gpuCullPerformance = {};
            std::fill(
                m_gpuCullTimestampValid.begin(),
                m_gpuCullTimestampValid.end(),
                uint8_t{0});
        }
        m_gpuCullDebugMode = mode;
    }

    GpuCullStats DX12Backend::gpuCullStats() const
    {
        return m_gpuCullStats;
    }

    GpuCullPerformance DX12Backend::gpuCullPerformance() const
    {
        return m_gpuCullPerformance;
    }

    D3D12_RESOURCE_STATES DX12Backend::usageToState(ResourceUsage usage) const
    {
        switch (usage)
        {
        case ResourceUsage::SampledTexture:
            return
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        case ResourceUsage::StorageTexture:
        case ResourceUsage::StorageBuffer:
            return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

        case ResourceUsage::IndirectArgument:
            return D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;

        case ResourceUsage::ColorAttachment:
            return D3D12_RESOURCE_STATE_RENDER_TARGET;

        case ResourceUsage::DepthAttachment:
            return D3D12_RESOURCE_STATE_DEPTH_WRITE;

        case ResourceUsage::VertexBuffer:
            return D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;

        case ResourceUsage::IndexBuffer:
            return D3D12_RESOURCE_STATE_INDEX_BUFFER;

        case ResourceUsage::ConstantBuffer:
            return D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;

        case ResourceUsage::TransferSrc:
            return D3D12_RESOURCE_STATE_COPY_SOURCE;

        case ResourceUsage::TransferDst:
            return D3D12_RESOURCE_STATE_COPY_DEST;

        case ResourceUsage::Present:
            return D3D12_RESOURCE_STATE_PRESENT;
        }

        return D3D12_RESOURCE_STATE_COMMON;
    }

    void DX12Backend::recreateSwapchain()
    {
        // Bump first so every path that recreates (reconcile, present failure)
        // advances the generation the renderer watches.
        ++m_swapchainGeneration;
        m_frameExecutor.waitForGpu();
        destroyDepthTarget();
        m_graphResourceRegistry.reset();
        destroyPathTraceResources();
        m_pipelineManager.shutdown();
        m_swapchain.resize();
        m_pipelineManager.init(m_device);
        m_gpuCullDiagnosticFrames = 0;
        m_gpuCullStats = {};
        m_gpuCullPerformance = {};
        std::fill(
            m_gpuCullTimestampValid.begin(),
            m_gpuCullTimestampValid.end(),
            uint8_t{0});
    }

    bool DX12Backend::globalIlluminationDiagnostics(
        GpuGiDiagnostics& result) const
    {
        std::memcpy(&result, m_giDiagnosticWords.data(), sizeof(result));
        return m_giDiagnosticFrames != 0;
    }

    void DX12Backend::setGlobalIlluminationDisplay(
        uint32_t debugView, float diagnosticIntensity, float debugExposure)
    {
        m_giDebugView = debugView;
        std::memcpy(&m_giDiagnosticIntensityBits, &diagnosticIntensity,
            sizeof(m_giDiagnosticIntensityBits));
        std::memcpy(&m_giDebugExposureBits, &debugExposure,
            sizeof(m_giDebugExposureBits));
    }

    void DX12Backend::setGlobalIlluminationRuntimeSettings(
        uint32_t maxSurfelUpdates, uint32_t maxProbeUpdates,
        uint32_t rayBudget, uint32_t freezeAfterFrames)
    {
        m_giMaxSurfelUpdates = std::max(maxSurfelUpdates, 1u);
        m_giMaxProbeUpdates = std::max(maxProbeUpdates, 64u);
        m_giRayBudget = std::max(rayBudget, 1u);
        m_giFreezeAfterFrames = freezeAfterFrames;
    }

    TextureFormat DX12Backend::swapchainTextureFormat() const
    {
        switch (m_swapchain.format())
        {
        case DXGI_FORMAT_R8G8B8A8_UNORM:
            return TextureFormat::RGBA8_UNorm;
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
            return TextureFormat::RGBA8_SRGB;
        case DXGI_FORMAT_B8G8R8A8_UNORM:
            return TextureFormat::BGRA8_UNorm;
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
            return TextureFormat::BGRA8_SRGB;
        default:
            return TextureFormat::RGBA8_UNorm;
        }
    }
}
