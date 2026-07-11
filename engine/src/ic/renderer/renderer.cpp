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
        bool graphDirty = true;
        RenderPathType pathType = RenderPathType::Forward;

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

        if (runtime.graphDirty)
        {
            buildOrRebuildFrameGraph();
        }

        syncSceneEnvironmentIBL(frame, scene);
        processPendingRendererJobs(frame);

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

        runtime.backend->execute(
            runtime.compiledGraphPlan,
            frame,
            scene);
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

        rt.path->buildFrameGraph(rctx, rt.builder);

        rt.compiledGraphPlan = rt.compiler.compile(rt.builder);

        rt.graphDirty = false;
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
        ImGui::Text("Passes %u | Levels %u | Batches %u",
            static_cast<uint32_t>(plan.nodes.size()),
            static_cast<uint32_t>(plan.executionLevels.size()),
            static_cast<uint32_t>(plan.queueSubmissions.size()));
        ImGui::TextDisabled(
            "CPU Parallel = recorded together. GPU Async = different queues may overlap.");

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

        if (ImGui::BeginTable("GraphQueueSchedule", 7,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
        {
            ImGui::TableSetupColumn("Level");
            ImGui::TableSetupColumn("Queue");
            ImGui::TableSetupColumn("Pass");
            ImGui::TableSetupColumn("Waits");
            ImGui::TableSetupColumn("CPU Parallel");
            ImGui::TableSetupColumn("GPU Async");
            ImGui::TableSetupColumn("Node");
            ImGui::TableHeadersRow();

            for (const QueueSubmissionBatch& batch : plan.queueSubmissions)
            {
                uint32_t batchesInLevel = 0;
                for (const QueueSubmissionBatch& other : plan.queueSubmissions)
                {
                    batchesInLevel +=
                        other.levelIndex == batch.levelIndex ? 1u : 0u;
                }

                const bool cpuParallel =
                    batch.levelIndex < plan.executionLevels.size() &&
                    plan.executionLevels[batch.levelIndex].nodeCount > 1;
                for (uint32_t i = 0; i < batch.nodeCount; ++i)
                {
                    const GraphNodeId node =
                        plan.queueSubmissionNodes[batch.firstNode + i];
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("%u", batch.levelIndex);
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(queueName(batch.queue));
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(passName(node));
                    ImGui::TableNextColumn();
                    ImGui::Text("%u", batch.waitCount);
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(cpuParallel ? "Yes" : "No");
                    ImGui::TableNextColumn();
                    if (batchesInLevel > 1)
                        ImGui::TextColored(
                            ImVec4(0.25f, 0.9f, 0.35f, 1.0f), "Eligible");
                    else
                        ImGui::TextDisabled("No");
                    ImGui::TableNextColumn();
                    ImGui::Text("%u", node);
                }
            }
            ImGui::EndTable();
        }
        ImGui::End();
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
