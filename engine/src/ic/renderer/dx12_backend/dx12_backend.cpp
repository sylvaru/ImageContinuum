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
#include "ic/renderer/renderer_specification.h"
#include "ic/renderer/path_tracing/path_trace_scene_builder.h"
#include "ic/renderer/renderer_common/renderer_util.h"

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <spdlog/spdlog.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_dx12.h>
#include <backends/imgui_impl_glfw.h>

#include <stdexcept>
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

        m_factory.init(spec.enableValidation);
        m_adapter.init(m_factory);
        m_device.init(m_adapter, m_factory.validationEnabled());
        m_resourceAllocator.init(m_device);

        const uint32_t framesInFlight =
            spec.framesInFlight == 0 ? 1 : spec.framesInFlight;

        const uint32_t workerSlots =
            workerCount == 0 ? 1 : workerCount;
        m_workerSlots = workerSlots;

        m_swapchain.setVsyncEnabled(spec.settings.vsync);

        m_swapchain.init(
            m_factory,
            m_device,
            window,
            framesInFlight);

        m_frameExecutor.init(m_device, m_swapchain, framesInFlight);

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
        m_gpuScene.init(
            m_device,
            m_resourceAllocator,
            m_descriptorSystem,
            framesInFlight);
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

        shutdownImGui();
        destroySceneResources();
        destroyEnvironmentResources();
        destroyClusteredForwardResources();
        destroyPathTraceResources();
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
        m_resourceStates.clear();

        spdlog::info("[DX12Backend] Shutdown");
    }

    void DX12Backend::execute(
        const CompiledGraphPlan& plan,
        const FrameContext& ctx,
        const SceneRenderView& scene)
    {
        if (!m_frameExecutor.ready())
        {
            return;
        }

        if (planUsesPathTracing(plan))
        {
            m_frameExecutor.waitForGpu();
        }

        const uint32_t frameSlot =
            static_cast<uint32_t>(
                ctx.frameIndex % m_frameExecutor.framesInFlight());

        m_frameExecutor.waitForFrame(frameSlot);

        if (!m_swapchain.updateSizeFromWindow())
        {
            recreateSwapchain();

            if (!m_swapchain.validForRendering())
            {
                return;
            }
        }

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

        std::vector<ID3D12CommandList*> commandLists;
        executeGraph(
            plan,
            ctx,
            scene,
            swapchainImage,
            commandLists);
        recordImGui(
            ctx,
            swapchainImage,
            commandLists);
        m_device.logValidationMessages();

        const bool presented =
            m_frameExecutor.submitAndPresent(plan, commandLists, frameSlot);

        if (!presented)
        {
            recreateSwapchain();
        }
    }


    void DX12Backend::executeGraph(
        const CompiledGraphPlan& plan,
        const FrameContext& ctx,
        const SceneRenderView& scene,
        ID3D12Resource* swapchainImage,
        std::vector<ID3D12CommandList*>& commandLists)
    {
        // Resolve lazy pipeline/resource state before worker threads begin.
        // Recording jobs may only read this shared renderer state.
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
                (void)computePipelineForNode(plan, node);
            }
        }

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

                applyBarriers(
                    cmd,
                    plan,
                    node,
                    swapchainImage);

                dispatchNode(
                    plan,
                    node,
                    ctx,
                    scene,
                    cmd,
                    swapchainImage);

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

                for (const ResourceLifetime& lifetime :
                     plan.resourceLifetimes)
                {
                    if (lifetime.lastUse != node.nodeId ||
                        lifetime.resource >= plan.resources.size() ||
                        plan.resources[lifetime.resource].ownership !=
                            ResourceOwnership::Transient)
                    {
                        continue;
                    }
                    DX12GraphResourceEntry* entry =
                        m_graphResourceRegistry.entry(lifetime.resource);
                    if (!entry || entry->state == D3D12_RESOURCE_STATE_COMMON)
                    {
                        continue;
                    }
                    ID3D12Resource* resource =
                        entry->type == GraphResourceType::Texture
                            ? entry->texture.resource.Get()
                            : entry->buffer.resource.Get();
                    transitionResource(
                        cmd, resource, entry->state,
                        D3D12_RESOURCE_STATE_COMMON);
                    entry->state = D3D12_RESOURCE_STATE_COMMON;
                }

                throwIfFailed(
                    cmd->Close(),
                    "Failed to close DX12 frame command list.");

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
        else if (DX12GraphResourceEntry* entry =
                     m_graphResourceRegistry.entry(barrier.resource))
        {
            dxResource = entry->type == GraphResourceType::Texture
                ? entry->texture.resource.Get()
                : entry->buffer.resource.Get();
        }

        if (!dxResource)
        {
            spdlog::error(
                "[DX12Backend] Missing graph resource handle for barrier resource {}",
                barrier.resource);
            return;
        }

        const D3D12_RESOURCE_STATES before =
            (crossQueueAcquire || barrier.firstUse)
            ? D3D12_RESOURCE_STATE_COMMON
            : (resource.ownership == ResourceOwnership::Transient &&
                m_graphResourceRegistry.entry(barrier.resource)
                ? m_graphResourceRegistry.entry(barrier.resource)->state
                : usageToState(barrier.oldUsage));

        D3D12_RESOURCE_STATES after = crossQueueRelease
            ? D3D12_RESOURCE_STATE_COMMON
            : usageToState(barrier.newUsage);
        if (!crossQueueRelease && commandQueue == QueueType::Compute &&
            barrier.newUsage == ResourceUsage::SampledTexture)
        {
            after = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        }

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
            return;
        }

        transitionResource(
            cmd,
            dxResource,
            before,
            after);

        if (resource.ownership == ResourceOwnership::Transient)
        {
            if (DX12GraphResourceEntry* entry =
                    m_graphResourceRegistry.entry(barrier.resource))
            {
                entry->state = after;
            }
        }

    }

    void DX12Backend::dispatchNode(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node,
        const FrameContext& ctx,
        const SceneRenderView& scene,
        ID3D12GraphicsCommandList4* cmd,
        ID3D12Resource* swapchainImage)
    {
        switch (node.type)
        {
        case GraphNodeType::Graphics:
            executeGraphicsNode(plan, node, ctx, scene, cmd, swapchainImage);
            break;

        case GraphNodeType::Compute:
            executeComputeNode(plan, node, ctx, scene, cmd);
            break;

        case GraphNodeType::Transfer:
            executeTransferNode(plan, node, ctx, cmd, swapchainImage);
            break;

        case GraphNodeType::Present:
            break;
        }
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
                colorEntry && colorEntry->ownership == ResourceOwnership::Transient &&
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
                depthEntry && depthEntry->ownership == ResourceOwnership::Transient &&
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
                1.0f,
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

        const std::span<const DX12GpuScene::DrawItem> draws = m_gpuScene.draws();

        const uint32_t frameSlot =
            static_cast<uint32_t>(ctx.frameIndex % m_gpuScene.frameSlotCount());
        DX12GpuSceneFrameResources& frameResources =
            m_gpuScene.frameResources(frameSlot);

        ID3D12DescriptorHeap* heaps[] =
        {
            m_descriptorSystem.shaderResourceHeap(),
            m_descriptorSystem.samplerHeap()
        };

        cmd->SetDescriptorHeaps(
            static_cast<UINT>(std::size(heaps)),
            heaps);

        cmd->SetGraphicsRootSignature(pipeline->rootSignature.Get());
        cmd->SetPipelineState(pipeline->pipelineState.Get());
        if (pipeline->desc.bindingLayout ==
            PipelineBindingLayoutKind::ClusteredForward)
        {
            bindClusteredForwardGraphics(*pipeline, ctx, cmd);
        }
        else
        {
            cmd->SetGraphicsRootConstantBufferView(
                0,
                frameResources.frameConstants.gpuAddress);
            cmd->SetGraphicsRootDescriptorTable(
                1,
                frameResources.objectSrv.gpuStart);
            cmd->SetGraphicsRootDescriptorTable(
                2,
                frameResources.materialSrv.gpuStart);
            cmd->SetGraphicsRootDescriptorTable(
                4,
                m_descriptorSystem.shaderResourceGpuStart());
            cmd->SetGraphicsRootDescriptorTable(
                5,
                m_descriptorSystem.samplerGpuStart());
            if (m_environmentResources.iblBaked)
            {
                cmd->SetGraphicsRootDescriptorTable(
                    6,
                    m_environmentResources.irradianceSrv.gpuStart);
                cmd->SetGraphicsRootDescriptorTable(
                    7,
                    m_environmentResources.prefilteredSrv.gpuStart);
                cmd->SetGraphicsRootDescriptorTable(
                    8,
                    m_environmentResources.brdfLutSrv.gpuStart);
                cmd->SetGraphicsRootDescriptorTable(
                    9,
                    m_environmentResources.sampler.gpuStart);
            }
            cmd->SetGraphicsRootShaderResourceView(
                10,
                m_gpuScene.drawMetadata.gpuAddress);
        }
        cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

        const DrawConstants unusedFallbackConstants{};
        cmd->SetGraphicsRoot32BitConstants(
            3,
            4,
            &unusedFallbackConstants,
            0);

        // DX12 command consumption remains behind this guard while the
        // command-signature/root-constant contract is validated on all
        // supported drivers. Culling and command generation stay active.
        constexpr bool enableDx12IndirectConsumption = false;
        const bool useGpuDriven =
            enableDx12IndirectConsumption &&
            m_gpuScene.indexedIndirectCommandSignature &&
            m_gpuScene.indirectArguments &&
            m_gpuScene.binCounts &&
            !m_gpuScene.geometryBins().empty();
        if (useGpuDriven)
        {
            if (m_gpuScene.indirectArgumentsState !=
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT)
            {
                transitionResource(cmd,
                    m_gpuScene.indirectArguments.resource.Get(),
                    m_gpuScene.indirectArgumentsState,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
                m_gpuScene.indirectArgumentsState =
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
            }
            if (m_gpuScene.binCountsState !=
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT)
            {
                transitionResource(cmd,
                    m_gpuScene.binCounts.resource.Get(),
                    m_gpuScene.binCountsState,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
                m_gpuScene.binCountsState =
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
            }
            if (m_gpuScene.drawMetadataState !=
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
            {
                transitionResource(cmd,
                    m_gpuScene.drawMetadata.resource.Get(),
                    m_gpuScene.drawMetadataState,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                m_gpuScene.drawMetadataState =
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            }
        }

        DX12IndirectDrawStream indirectStream{};
        indirectStream.commandSignature =
            m_gpuScene.indexedIndirectCommandSignature.Get();
        indirectStream.indirectArguments =
            m_gpuScene.indirectArguments.resource.Get();
        indirectStream.binCounts = m_gpuScene.binCounts.resource.Get();

        // Shared by the depth prepass and the forward pass: both pipelines
        // route scene geometry through this same recorder (see
        // dx12_pass_recorders.h for why there is no separate depth-only path).
        recordSceneGeometryDraws(
            cmd,
            draws,
            m_gpuScene.geometryBins(),
            useGpuDriven,
            indirectStream,
            [this](AssetHandle handle) -> DX12UploadedModel*
            {
                auto it = m_uploadedModels.find(handle);
                return it != m_uploadedModels.end() ? &it->second : nullptr;
            });
    }

    void DX12Backend::executeComputeNode(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node,
        [[maybe_unused]] const FrameContext& ctx,
        [[maybe_unused]] const SceneRenderView& scene,
        [[maybe_unused]] ID3D12GraphicsCommandList4* cmd)
    {
        if (node.payloadIndex >= plan.payloads.size())
        {
            return;
        }

        if (std::get_if<PathTracePassData>(&plan.payloads[node.payloadIndex]))
        {
            executePathTraceNode(plan, node, ctx, scene, cmd);
            return;
        }

        if (std::get_if<EnvironmentConvertPassData>(
                &plan.payloads[node.payloadIndex]))
        {
            executeEnvironmentConvertNode(plan, node, ctx, scene, cmd);
            return;
        }

        if (std::get_if<TonemapPassData>(&plan.payloads[node.payloadIndex]))
        {
            executeTonemapNode(plan, node, ctx, cmd);
            return;
        }

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

            cmd->SetComputeRootUnorderedAccessView(
                0,
                m_computeTestBuffer.gpuAddress);
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

            DX12CullBuffers cull{};
            cull.visibleInstances = g.visibleInstances.resource.Get();
            cull.visibleInstancesState = &g.visibleInstancesState;
            cull.visibleInstancesAddr = g.visibleInstances.gpuAddress;
            cull.visibleInstanceCount = g.visibleInstanceCount.resource.Get();
            cull.visibleInstanceCountState = &g.visibleInstanceCountState;
            cull.visibleInstanceCountAddr = g.visibleInstanceCount.gpuAddress;
            cull.indirectArguments = g.indirectArguments.resource.Get();
            cull.indirectArgumentsState = &g.indirectArgumentsState;
            cull.indirectArgumentsAddr = g.indirectArguments.gpuAddress;
            cull.drawMetadata = g.drawMetadata.resource.Get();
            cull.drawMetadataState = &g.drawMetadataState;
            cull.drawMetadataAddr = g.drawMetadata.gpuAddress;
            cull.binCounts = g.binCounts.resource.Get();
            cull.binCountsState = &g.binCountsState;
            cull.binCountsAddr = g.binCounts.gpuAddress;
            cull.frameConstantsAddr = frameResources.frameConstants.gpuAddress;
            cull.instanceBoundsAddr = frameResources.instanceBounds.gpuAddress;
            cull.drawInputsAddr = frameResources.drawInputs.gpuAddress;

            recordGpuFrustumCull(passCtx, cull);

            if (!g.loggedGpuCull)
            {
                spdlog::info(
                    "[DX12Backend] GPU frustum culling dispatch prepared for {} instance(s)",
                    m_gpuScene.instanceCount());
                g.loggedGpuCull = true;
            }
        }
        else if (pipeline->desc.bindingLayout ==
            PipelineBindingLayoutKind::ClusteredForward)
        {
            if (!bindClusteredForwardCompute(*pipeline, ctx, scene, cmd))
            {
                return;
            }
        }

        if (pipeline->desc.bindingLayout ==
            PipelineBindingLayoutKind::GpuFrustumCull)
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

        if (pipeline->desc.bindingLayout ==
            PipelineBindingLayoutKind::ComputeStorageBuffer)
        {
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            barrier.UAV.pResource = m_computeTestBuffer.resource.Get();
            cmd->ResourceBarrier(1, &barrier);
        }
        else if (pipeline->desc.bindingLayout ==
            PipelineBindingLayoutKind::ClusteredForward)
        {
            D3D12_RESOURCE_BARRIER barriers[4]{};
            ID3D12Resource* resources[] =
            {
                m_clusteredForwardResources.clusterBounds.resource.Get(),
                m_clusteredForwardResources.clusterLightGrid.resource.Get(),
                m_clusteredForwardResources.clusterLightIndices.resource.Get(),
                m_clusteredForwardResources.clusterLightCounter.resource.Get()
            };
            for (uint32_t i = 0; i < static_cast<uint32_t>(std::size(barriers)); ++i)
            {
                barriers[i].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                barriers[i].UAV.pResource = resources[i];
            }
            cmd->ResourceBarrier(
                static_cast<UINT>(std::size(barriers)),
                barriers);
        }
        else if (pipeline->desc.bindingLayout ==
            PipelineBindingLayoutKind::GpuFrustumCull)
        {
            D3D12_RESOURCE_BARRIER barriers[5]{};
            barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            barriers[0].UAV.pResource =
                m_gpuScene.visibleInstances.resource.Get();
            barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            barriers[1].UAV.pResource =
                m_gpuScene.visibleInstanceCount.resource.Get();
            barriers[2].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            barriers[2].UAV.pResource =
                m_gpuScene.indirectArguments.resource.Get();
            barriers[3].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            barriers[3].UAV.pResource =
                m_gpuScene.drawMetadata.resource.Get();
            barriers[4].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            barriers[4].UAV.pResource =
                m_gpuScene.binCounts.resource.Get();
            cmd->ResourceBarrier(5, barriers);
            readbackVisibleInstanceCount(ctx, cmd);
        }
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
        auto it = m_computePipelineHandles.find(pipelineId);
        ComputePipelineHandle handle{};
        if (it != m_computePipelineHandles.end())
        {
            handle = it->second;
        }
        else
        {
            ComputePipelineDesc desc =
                m_pipelineLibrary->resolveCompute(
                    pipelineId,
                    RendererBackendType::DX12);
            handle = m_pipelineManager.requestComputePipeline(desc);
            m_computePipelineHandles.emplace(pipelineId, handle);
        }

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

        ID3D12DescriptorHeap* heaps[] =
        {
            m_descriptorSystem.shaderResourceHeap(),
            m_descriptorSystem.samplerHeap()
        };
        cmd->SetDescriptorHeaps(
            static_cast<UINT>(std::size(heaps)),
            heaps);
        cmd->SetComputeRootSignature(pipeline.rootSignature.Get());
        if (!pipeline.pipelineState.Get())
        {
            spdlog::error(
                "[DX12] Environment conversion PSO is null; skipping HDR conversion");
            return false;
        }
        cmd->SetPipelineState(pipeline.pipelineState.Get());
        cmd->SetComputeRootDescriptorTable(0, textureIt->second.srv.gpuStart);
        cmd->SetComputeRootDescriptorTable(1, m_environmentResources.cubemapUav.gpuStart);
        cmd->SetComputeRootDescriptorTable(2, m_environmentResources.sampler.gpuStart);
        cmd->Dispatch(
            (m_environmentResources.cubemapSize + 7u) / 8u,
            (m_environmentResources.cubemapSize + 7u) / 8u,
            6u);

        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource =
            m_environmentResources.cubemap.resource.Get();
        cmd->ResourceBarrier(1, &barrier);

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

        PathTraceConstants constants{};
        constants.renderWidth = m_pathTraceResources.width;
        constants.renderHeight = m_pathTraceResources.height;
        constants.frameIndex = static_cast<uint32_t>(ctx.frameIndex);
        constants.accumulatedSampleCount =
            m_pathTraceResources.accumulatedSampleCount;
        constants.exposure = m_pathTraceResources.tonemapExposure;
        constants.resetAccumulation =
            m_pathTraceResources.resetAccumulation ? 1u : 0u;
        constants.maxBounces = DefaultPathTraceMaxBounces;
        constants.samplesPerPixel = DefaultPathTraceSamplesPerPixel;
        constants.sceneVertexCount =
            m_pathTraceResources.sceneVertexCount;
        constants.sceneMaterialCount =
            m_pathTraceResources.sceneMaterialCount;
        constants.sceneTriangleCount =
            m_pathTraceResources.sceneTriangleCount;
        constants.sceneBvhNodeCount =
            m_pathTraceResources.sceneBvhNodeCount;
        constants.sceneEmissiveTriangleIndex =
            m_pathTraceResources.firstEmissiveTriangleIndex;
        constants.useSceneGeometry =
            m_pathTraceResources.sceneTriangleCount != 0u &&
            m_pathTraceResources.sceneBvhNodeCount != 0u
                ? 1u
                : 0u;
        const bool environmentResourcesAvailable =
            ensureEnvironmentResources(ctx, scene, cmd);
        if (!environmentResourcesAvailable)
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
        const bool environmentReady = m_environmentResources.converted;

        constants.environmentEnabled = environmentReady ? 1u : 0u;
        constants.environmentIntensity = scene.environment.settings.intensity;
        constants.environmentExposure =
            scene.environment.settings.pathTraceExposure;
        fillPathTraceCameraConstants(
            scene.camera,
            m_pathTraceResources.width,
            m_pathTraceResources.height,
            constants);
        for (const SceneLightRenderItem& light : scene.lights)
        {
            if (light.type != LightType::Point ||
                constants.pointLightCount >= MaxPathTracePointLights)
            {
                continue;
            }

            const uint32_t lightIndex = constants.pointLightCount++;
            constants.pointLightPositionRange[lightIndex] =
                glm::vec4(light.position, light.range);
            constants.pointLightColorIntensity[lightIndex] =
                glm::vec4(light.color, light.intensity);
        }

        const uint32_t frameSlot =
            static_cast<uint32_t>(
                ctx.frameIndex %
                m_pathTraceResources.pathTraceConstants.size());
        DX12Buffer& pathTraceConstants =
            m_pathTraceResources.pathTraceConstants[frameSlot];

        std::memcpy(
            pathTraceConstants.mapped,
            &constants,
            sizeof(constants));

        ID3D12DescriptorHeap* heaps[] =
        {
            m_descriptorSystem.shaderResourceHeap(),
            m_descriptorSystem.samplerHeap()
        };
        cmd->SetDescriptorHeaps(
            static_cast<UINT>(std::size(heaps)),
            heaps);

        DX12ComputePipeline* pipeline =
            m_pipelineManager.computePipeline(pathTracePipelineHandle);
        if (!pipeline)
        {
            return;
        }
        if (!pipeline->rootSignature || !pipeline->pipelineState)
        {
            spdlog::error(
                "[DX12] Path trace pipeline is incomplete; skipping dispatch");
            return;
        }
        cmd->SetComputeRootSignature(pipeline->rootSignature.Get());
        cmd->SetPipelineState(pipeline->pipelineState.Get());
        cmd->SetComputeRootConstantBufferView(
            0,
            pathTraceConstants.gpuAddress);
        cmd->SetComputeRootDescriptorTable(
            1,
            m_pathTraceResources.accumulationUav.gpuStart);
        if (m_pathTraceResources.sceneSrvs.valid())
        {
            if (m_environmentResources.cubemapSrv.valid() &&
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
            cmd->SetComputeRootDescriptorTable(
                2,
                m_pathTraceResources.sceneSrvs.gpuStart);
        }
        if (m_environmentResources.sampler.valid())
        {
            cmd->SetComputeRootDescriptorTable(
                3,
                m_environmentResources.sampler.gpuStart);
        }
        cmd->SetComputeRootDescriptorTable(
            4,
            m_descriptorSystem.shaderResourceGpuStart());
        cmd->SetComputeRootDescriptorTable(
            5,
            m_descriptorSystem.samplerGpuStart());

        const uint32_t groupCountX =
            (m_pathTraceResources.width + 7u) / 8u;
        const uint32_t groupCountY =
            (m_pathTraceResources.height + 7u) / 8u;
        cmd->Dispatch(groupCountX, groupCountY, 1);

        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource =
            m_pathTraceResources.accumulation.resource.Get();
        cmd->ResourceBarrier(1, &barrier);

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

        TonemapConstants constants{};
        constants.renderWidth = m_pathTraceResources.width;
        constants.renderHeight = m_pathTraceResources.height;
        constants.exposure = m_pathTraceResources.tonemapExposure;

        const uint32_t frameSlot =
            static_cast<uint32_t>(
                ctx.frameIndex %
                m_pathTraceResources.tonemapConstants.size());
        DX12Buffer& tonemapConstants =
            m_pathTraceResources.tonemapConstants[frameSlot];

        std::memcpy(
            tonemapConstants.mapped,
            &constants,
            sizeof(constants));

        ID3D12DescriptorHeap* heaps[] =
        {
            m_descriptorSystem.shaderResourceHeap()
        };
        cmd->SetDescriptorHeaps(1, heaps);
        cmd->SetComputeRootSignature(pipeline->rootSignature.Get());
        cmd->SetPipelineState(pipeline->pipelineState.Get());
        cmd->SetComputeRootConstantBufferView(
            0,
            tonemapConstants.gpuAddress);
        cmd->SetComputeRootDescriptorTable(
            1,
            m_pathTraceResources.tonemapUav.gpuStart);
        cmd->SetComputeRootDescriptorTable(
            2,
            m_pathTraceResources.accumulationSrv.gpuStart);

        const uint32_t groupCountX =
            (m_pathTraceResources.width + 7u) / 8u;
        const uint32_t groupCountY =
            (m_pathTraceResources.height + 7u) / 8u;
        cmd->Dispatch(groupCountX, groupCountY, 1);

        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource =
            m_pathTraceResources.tonemap.resource.Get();
        cmd->ResourceBarrier(1, &barrier);
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

        if (pass->name == "PathTracer.CopyToBackBuffer" &&
            m_pathTraceResources.tonemap &&
            swapchainImage)
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

            cmd->CopyResource(
                swapchainImage,
                m_pathTraceResources.tonemap.resource.Get());
        }
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

        auto it = m_pipelineHandles.find(pass->pipeline);
        if (it != m_pipelineHandles.end())
        {
            return it->second;
        }

        GraphicsPipelineDesc desc =
            m_pipelineLibrary->resolveGraphics(
                pass->pipeline,
                RendererBackendType::DX12,
                swapchainTextureFormat());

        GraphicsPipelineHandle handle =
            m_pipelineManager.requestGraphicsPipeline(desc);
        m_pipelineHandles.emplace(pass->pipeline, handle);
        return handle;
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

        auto it = m_computePipelineHandles.find(pipelineId);
        if (it != m_computePipelineHandles.end())
        {
            return it->second;
        }

        ComputePipelineDesc desc =
            m_pipelineLibrary->resolveCompute(
                pipelineId,
                RendererBackendType::DX12);

        ComputePipelineHandle handle =
            m_pipelineManager.requestComputePipeline(desc);
        m_computePipelineHandles.emplace(pipelineId, handle);
        return handle;
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

        m_pipelineHandles.clear();
        m_computePipelineHandles.clear();
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
        PathTraceSceneData sceneData =
            buildPathTraceSceneData(
                scene,
                assets,
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
                });
        if (sceneData.triangles.empty() &&
            m_pathTraceResources.sceneSrvs.valid())
        {
            return;
        }

        uploadPathTraceScene(sceneData);

        m_pathTraceResources.sceneVersion = scene.sceneVersion;
        m_pathTraceResources.sceneHadPendingModels = hasPendingModels;
        m_pathTraceResources.accumulatedSampleCount = 0;
        m_pathTraceResources.resetAccumulation = true;
    }

    void DX12Backend::uploadPathTraceScene(
        const PathTraceSceneData& sceneData)
    {
        if (m_pathTraceResources.sceneVertices ||
            m_pathTraceResources.sceneMaterials ||
            m_pathTraceResources.sceneTriangles ||
            m_pathTraceResources.sceneBvhNodes)
        {
            m_frameExecutor.waitForGpu();
        }

        destroyPathTraceSceneResources();

        m_pathTraceResources.sceneVertexCount =
            static_cast<uint32_t>(sceneData.vertices.size());
        m_pathTraceResources.sceneMaterialCount =
            static_cast<uint32_t>(sceneData.materials.size());
        m_pathTraceResources.sceneTriangleCount =
            static_cast<uint32_t>(sceneData.triangles.size());
        m_pathTraceResources.sceneBvhNodeCount =
            static_cast<uint32_t>(sceneData.bvhNodes.size());
        m_pathTraceResources.firstEmissiveTriangleIndex =
            sceneData.firstEmissiveTriangleIndex;

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
                sceneData.triangles.data(),
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
            Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
            throwIfFailed(
                m_device.device()->CreateCommandAllocator(
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    IID_PPV_ARGS(&allocator)),
                "Failed to create DX12 path trace upload allocator.");

            Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> cmd;
            throwIfFailed(
                m_device.device()->CreateCommandList(
                    0,
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    allocator.Get(),
                    nullptr,
                    IID_PPV_ARGS(&cmd)),
                "Failed to create DX12 path trace upload command list.");

            for (PendingSceneUpload& upload : pendingUploads)
            {
                transitionResource(
                    cmd.Get(),
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
                    cmd.Get(),
                    upload.destination->resource.Get(),
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }

            throwIfFailed(
                cmd->Close(),
                "Failed to close DX12 path trace upload command list.");

            ID3D12CommandList* lists[] = { cmd.Get() };
            m_device.graphicsQueue()->ExecuteCommandLists(1, lists);
            m_frameExecutor.waitForGpu();

            for (PendingSceneUpload& upload : pendingUploads)
            {
                m_resourceAllocator.destroyBuffer(upload.staging);
            }
        }

        m_descriptorSystem.releaseResourceDescriptors(
            m_pathTraceResources.sceneSrvs);
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

        if (m_clusteredForwardResources.clusterBounds &&
            m_clusteredForwardResources.width == width &&
            m_clusteredForwardResources.height == height)
        {
            return;
        }

        destroyClusteredForwardResources();
        m_gpuScene.destroyCullBuffers();

        m_clusteredForwardResources.width = width;
        m_clusteredForwardResources.height = height;
        m_clusteredForwardResources.clusterCountX = clusterCountX;
        m_clusteredForwardResources.clusterCountY = clusterCountY;
        m_clusteredForwardResources.clusterCountZ = clusterCountZ;
        m_clusteredForwardResources.clusterCount = clusterCount;
        m_clusteredForwardResources.hiZMipCount = hiZMipCount;

        const uint64_t maxClusterLightRefs =
            static_cast<uint64_t>(clusterCount) *
            ClusteredForwardMaxLightsPerCluster;

        m_clusteredForwardResources.clusterBounds =
            m_resourceAllocator.createBuffer({
                .size = clusterCount * sizeof(GpuClusterBounds),
                .usage = BufferUsageFlags::Storage,
                .memoryUsage = ResourceMemoryUsage::GpuOnly,
                .debugName = "DX12 clustered cluster bounds"
            });
        m_clusteredForwardResources.clusterLightGrid =
            m_resourceAllocator.createBuffer({
                .size = clusterCount * sizeof(GpuClusterLightGrid),
                .usage = BufferUsageFlags::Storage,
                .memoryUsage = ResourceMemoryUsage::GpuOnly,
                .debugName = "DX12 clustered light grid"
            });
        m_clusteredForwardResources.clusterLightIndices =
            m_resourceAllocator.createBuffer({
                .size = maxClusterLightRefs * sizeof(uint32_t),
                .usage = BufferUsageFlags::Storage,
                .memoryUsage = ResourceMemoryUsage::GpuOnly,
                .debugName = "DX12 clustered light indices"
            });
        m_clusteredForwardResources.clusterLightCounter =
            m_resourceAllocator.createBuffer({
                .size = sizeof(uint32_t),
                .usage = BufferUsageFlags::Storage,
                .memoryUsage = ResourceMemoryUsage::GpuOnly,
                .debugName = "DX12 clustered light counter"
            });
        m_gpuScene.ensureCullBuffers(
            m_device.device(), maxInstances, MaxGpuDrivenBins);

        m_clusteredForwardResources.boundsState =
            m_clusteredForwardResources.clusterBounds.initialState;
        m_clusteredForwardResources.gridState =
            m_clusteredForwardResources.clusterLightGrid.initialState;
        m_clusteredForwardResources.indicesState =
            m_clusteredForwardResources.clusterLightIndices.initialState;
        m_clusteredForwardResources.counterState =
            m_clusteredForwardResources.clusterLightCounter.initialState;

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

    void DX12Backend::readbackVisibleInstanceCount(
        const FrameContext& ctx,
        ID3D12GraphicsCommandList4* cmd)
    {
        if (!m_gpuScene.visibleInstanceCount ||
            !m_gpuScene.visibleInstanceCountReadback)
        {
            return;
        }
        if (!m_hiZDebugViewEnabled || (ctx.frameIndex % 30u) != 0u)
        {
            return;
        }

        transitionResource(
            cmd,
            m_gpuScene.visibleInstanceCount.resource.Get(),
            m_gpuScene.visibleInstanceCountState,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
        m_gpuScene.visibleInstanceCountState =
            D3D12_RESOURCE_STATE_COPY_SOURCE;
        cmd->CopyBufferRegion(
            m_gpuScene.visibleInstanceCountReadback.resource.Get(),
            0,
            m_gpuScene.visibleInstanceCount.resource.Get(),
            0,
            sizeof(uint32_t));

        if (m_gpuScene.visibleInstanceCountReadback.mapped)
        {
            const uint32_t visible =
                *static_cast<const uint32_t*>(
                    m_gpuScene.visibleInstanceCountReadback.mapped);
            if (visible != m_gpuScene.lastVisibleInstanceCount)
            {
                spdlog::info(
                    "[DX12Backend] GPU frustum visible instances: {} -> {}",
                    m_gpuScene.lastVisibleInstanceCount,
                    visible);
                m_gpuScene.lastVisibleInstanceCount = visible;
            }
        }
    }

    bool DX12Backend::bindClusteredForwardCompute(
        const DX12ComputePipeline& pipeline,
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

        auto transitionClusterBuffer =
            [&](DX12Buffer& buffer, D3D12_RESOURCE_STATES& state)
            {
                if (state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
                {
                    transitionResource(
                        cmd,
                        buffer.resource.Get(),
                        state,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                }
            };
        transitionClusterBuffer(
            m_clusteredForwardResources.clusterBounds,
            m_clusteredForwardResources.boundsState);
        transitionClusterBuffer(
            m_clusteredForwardResources.clusterLightGrid,
            m_clusteredForwardResources.gridState);
        transitionClusterBuffer(
            m_clusteredForwardResources.clusterLightIndices,
            m_clusteredForwardResources.indicesState);
        transitionClusterBuffer(
            m_clusteredForwardResources.clusterLightCounter,
            m_clusteredForwardResources.counterState);

        ID3D12DescriptorHeap* heaps[] =
        {
            m_descriptorSystem.shaderResourceHeap(),
            m_descriptorSystem.samplerHeap()
        };
        cmd->SetDescriptorHeaps(
            static_cast<UINT>(std::size(heaps)),
            heaps);
        cmd->SetComputeRootSignature(pipeline.rootSignature.Get());
        cmd->SetComputeRootConstantBufferView(
            0,
            frameResources.frameConstants.gpuAddress);

        cmd->SetComputeRootUnorderedAccessView(
            10,
            m_clusteredForwardResources.clusterBounds.gpuAddress);

        cmd->SetComputeRootUnorderedAccessView(
            11,
            m_clusteredForwardResources.clusterLightGrid.gpuAddress);

        cmd->SetComputeRootUnorderedAccessView(
            12,
            m_clusteredForwardResources.clusterLightIndices.gpuAddress);

        cmd->SetComputeRootShaderResourceView(
            13,
            frameResources.visibleLights.gpuAddress);

        cmd->SetComputeRootUnorderedAccessView(
            14,
            m_clusteredForwardResources.clusterLightCounter.gpuAddress);
        return true;
    }

    void DX12Backend::bindClusteredForwardGraphics(
        const DX12GraphicsPipeline& pipeline,
        const FrameContext& ctx,
        ID3D12GraphicsCommandList4* cmd)
    {
        const uint32_t frameSlot =
            static_cast<uint32_t>(ctx.frameIndex % m_gpuScene.frameSlotCount());
        DX12GpuSceneFrameResources& frameResources =
            m_gpuScene.frameResources(frameSlot);

        cmd->SetGraphicsRootSignature(pipeline.rootSignature.Get());
        cmd->SetGraphicsRootConstantBufferView(
            0,
            frameResources.frameConstants.gpuAddress);
        cmd->SetGraphicsRootDescriptorTable(1, frameResources.objectSrv.gpuStart);
        cmd->SetGraphicsRootDescriptorTable(2, frameResources.materialSrv.gpuStart);
        cmd->SetGraphicsRootDescriptorTable(
            4,
            m_descriptorSystem.shaderResourceGpuStart());

        cmd->SetGraphicsRootDescriptorTable(
            5,
            m_descriptorSystem.samplerGpuStart());

        if (m_environmentResources.iblBaked)
        {
            cmd->SetGraphicsRootDescriptorTable(
                6,
                m_environmentResources.irradianceSrv.gpuStart);
            cmd->SetGraphicsRootDescriptorTable(
                7,
                m_environmentResources.prefilteredSrv.gpuStart);
            cmd->SetGraphicsRootDescriptorTable(
                8,
                m_environmentResources.brdfLutSrv.gpuStart);
            cmd->SetGraphicsRootDescriptorTable(
                9,
                m_environmentResources.sampler.gpuStart);
        }
        cmd->SetGraphicsRootUnorderedAccessView(
            10,
            m_clusteredForwardResources.clusterBounds.gpuAddress);
        cmd->SetGraphicsRootUnorderedAccessView(
            11,
            m_clusteredForwardResources.clusterLightGrid.gpuAddress);
        cmd->SetGraphicsRootUnorderedAccessView(
            12,
            m_clusteredForwardResources.clusterLightIndices.gpuAddress);
        cmd->SetGraphicsRootShaderResourceView(
            13,
            frameResources.visibleLights.gpuAddress);
        cmd->SetGraphicsRootUnorderedAccessView(
            14,
            m_clusteredForwardResources.clusterLightCounter.gpuAddress);
        cmd->SetGraphicsRootShaderResourceView(
            15,
            m_clusteredForwardResources.clusterBounds.gpuAddress);
        cmd->SetGraphicsRootShaderResourceView(
            16,
            m_clusteredForwardResources.clusterLightGrid.gpuAddress);
        cmd->SetGraphicsRootShaderResourceView(
            17,
            m_clusteredForwardResources.clusterLightIndices.gpuAddress);
        cmd->SetGraphicsRootShaderResourceView(
            18,
            m_gpuScene.drawMetadata.gpuAddress);
    }

    void DX12Backend::destroyClusteredForwardResources()
    {
        m_resourceAllocator.destroyBuffer(
            m_clusteredForwardResources.clusterBounds);
        m_resourceAllocator.destroyBuffer(
            m_clusteredForwardResources.clusterLightGrid);
        m_resourceAllocator.destroyBuffer(
            m_clusteredForwardResources.clusterLightIndices);
        m_resourceAllocator.destroyBuffer(
            m_clusteredForwardResources.clusterLightCounter);
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
            const uint32_t frameCount =
                std::max<uint32_t>(1u, m_workerSlots);
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

        SkyboxConstants constants{};
        fillSkyboxConstants(
            scene.camera,
            m_swapchain.width(),
            m_swapchain.height(),
            scene.environment.settings,
            constants);
        std::memcpy(
            constantsBuffer.mapped,
            &constants,
            sizeof(constants));

        ID3D12DescriptorHeap* heaps[] =
        {
            m_descriptorSystem.shaderResourceHeap(),
            m_descriptorSystem.samplerHeap()
        };
        cmd->SetDescriptorHeaps(
            static_cast<UINT>(std::size(heaps)),
            heaps);
        cmd->SetGraphicsRootSignature(pipeline.rootSignature.Get());
        cmd->SetPipelineState(pipeline.pipelineState.Get());
        cmd->SetGraphicsRootConstantBufferView(
            0,
            constantsBuffer.gpuAddress);
        cmd->SetGraphicsRootDescriptorTable(
            1,
            m_environmentResources.cubemapSrv.gpuStart);
        cmd->SetGraphicsRootDescriptorTable(
            2,
            m_environmentResources.sampler.gpuStart);
        cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        cmd->DrawInstanced(3, 1, 0, 0);
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
                    auto it = m_computePipelineHandles.find(pipelineId);
                    ComputePipelineHandle handle{};
                    if (it != m_computePipelineHandles.end())
                    {
                        handle = it->second;
                    }
                    else
                    {
                        ComputePipelineDesc desc =
                            m_pipelineLibrary->resolveCompute(
                                pipelineId,
                                RendererBackendType::DX12);
                        handle = m_pipelineManager.requestComputePipeline(desc);
                        m_computePipelineHandles.emplace(pipelineId, handle);
                    }
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

            convertPipeline =
                m_pipelineManager.computePipeline(
                    m_computePipelineHandles.at(
                        makePipelineId("equirect_to_cubemap")));
            irradiancePipeline =
                m_pipelineManager.computePipeline(
                    m_computePipelineHandles.at(
                        makePipelineId("ibl_irradiance")));
            prefilterPipeline =
                m_pipelineManager.computePipeline(
                    m_computePipelineHandles.at(
                        makePipelineId("ibl_prefilter")));
            brdfPipeline =
                m_pipelineManager.computePipeline(
                    m_computePipelineHandles.at(
                        makePipelineId("ibl_brdf_lut")));

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
                (void)requestTexture(
                    request.desc.sourceEnvironment,
                    0,
                    *image,
                    TextureTransferFunction::Linear);

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
        TextureTransferFunction transfer)
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

        UploadedTexture uploaded{};
        uploaded.texture =
            m_resourceAllocator.createTexture(
                uploadImage,
                TextureUsageFlags::Sampled | TextureUsageFlags::TransferDst,
                "DX12 bindless model texture");
        uploaded.srv =
            m_descriptorSystem.allocateResourceDescriptors(1);

        const uint64_t sourceBytes = imageByteSize(uploadImage);

        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        UINT rowCount = 0;
        UINT64 rowSize = 0;
        UINT64 uploadBytes = 0;
        m_device.device()->GetCopyableFootprints(
            &uploaded.texture.desc,
            0,
            1,
            0,
            &footprint,
            &rowCount,
            &rowSize,
            &uploadBytes);

        DX12Buffer staging =
            m_resourceAllocator.createBuffer({
                .size = uploadBytes,
                .usage = BufferUsageFlags::TransferSrc,
                .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                .mappedAtCreation = true,
                .debugName = "DX12 texture upload staging"
            });

        const uint8_t* src =
            reinterpret_cast<const uint8_t*>(uploadImage.pixels.data());
        uint8_t* dst =
            reinterpret_cast<uint8_t*>(staging.mapped) + footprint.Offset;
        const uint64_t tightRowBytes =
            uploadImage.height != 0 ? sourceBytes / uploadImage.height : 0;
        for (uint32_t row = 0; row < uploadImage.height; ++row)
        {
            std::memcpy(
                dst + static_cast<uint64_t>(row) * footprint.Footprint.RowPitch,
                src + static_cast<uint64_t>(row) * tightRowBytes,
                static_cast<size_t>(tightRowBytes));
        }

        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
        throwIfFailed(
            m_device.device()->CreateCommandAllocator(
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                IID_PPV_ARGS(&allocator)),
            "Failed to create DX12 texture upload allocator.");

        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> cmd;
        throwIfFailed(
            m_device.device()->CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                allocator.Get(),
                nullptr,
                IID_PPV_ARGS(&cmd)),
            "Failed to create DX12 texture upload command list.");

        transitionResource(
            cmd.Get(),
            uploaded.texture.resource.Get(),
            uploaded.texture.initialState,
            D3D12_RESOURCE_STATE_COPY_DEST);

        D3D12_TEXTURE_COPY_LOCATION dstLocation{};
        dstLocation.pResource = uploaded.texture.resource.Get();
        dstLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;

        D3D12_TEXTURE_COPY_LOCATION srcLocation{};
        srcLocation.pResource = staging.resource.Get();
        srcLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        srcLocation.PlacedFootprint = footprint;

        cmd->CopyTextureRegion(&dstLocation, 0, 0, 0, &srcLocation, nullptr);

        transitionResource(
            cmd.Get(),
            uploaded.texture.resource.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        uploaded.state =
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        throwIfFailed(
            cmd->Close(),
            "Failed to close DX12 texture upload command list.");

        ID3D12CommandList* lists[] = { cmd.Get() };
        m_device.graphicsQueue()->ExecuteCommandLists(1, lists);
        m_frameExecutor.waitForGpu();

        m_resourceAllocator.destroyBuffer(staging);

        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Shader4ComponentMapping =
            D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Format = uploaded.texture.desc.Format;
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Texture2D.MipLevels = 1;

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
        desc.Filter = sampler
            ? toFilter(sampler->minFilter, sampler->magFilter)
            : D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        desc.AddressU = sampler ? toAddress(sampler->wrapU) : D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        desc.AddressV = sampler ? toAddress(sampler->wrapV) : D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
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
                    BufferUsageFlags::TransferDst,
                .memoryUsage = ResourceMemoryUsage::GpuOnly,
                .mappedAtCreation = false,
                .debugName = "DX12 model index buffer"
            });

        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
        throwIfFailed(
            m_device.device()->CreateCommandAllocator(
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                IID_PPV_ARGS(&allocator)),
            "Failed to create DX12 model upload allocator.");

        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> cmd;
        throwIfFailed(
            m_device.device()->CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                allocator.Get(),
                nullptr,
                IID_PPV_ARGS(&cmd)),
            "Failed to create DX12 model upload command list.");

        transitionResource(
            cmd.Get(),
            uploaded.vertexBuffer.resource.Get(),
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_DEST);
        transitionResource(
            cmd.Get(),
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
            cmd.Get(),
            uploaded.vertexBuffer.resource.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
        transitionResource(
            cmd.Get(),
            uploaded.indexBuffer.resource.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_INDEX_BUFFER);

        throwIfFailed(
            cmd->Close(),
            "Failed to close DX12 model upload command list.");

        ID3D12CommandList* lists[] = { cmd.Get() };
        m_device.graphicsQueue()->ExecuteCommandLists(1, lists);
        m_frameExecutor.waitForGpu();

        m_resourceAllocator.destroyBuffer(vertexStaging);
        m_resourceAllocator.destroyBuffer(indexStaging);

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

        return m_gpuScene.prepare(
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
                frameData.projection = glm::perspectiveRH_ZO(
                    scene.camera.verticalFovRadians,
                    scene.camera.aspectRatio,
                    scene.camera.nearPlane,
                    scene.camera.farPlane);
                frameData.viewProjection = frameData.projection * frameData.view;
                frameData.cameraPosition = scene.camera.position;
                frameData.time = ctx.timeSinceStart;
                frameData.environmentEnabled =
                    m_environmentResources.iblBaked ? 1u : 0u;
                frameData.prefilteredMipCount =
                    m_environmentResources.prefilterMipCount;
                frameData.environmentIntensity =
                    scene.environment.settings.intensity;
                frameData.environmentExposure =
                    scene.environment.settings.skyboxExposure;

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
                    0u);
                frameData.cullingConfig = glm::uvec4(
                    std::min<uint32_t>(
                        instanceBoundsCount, ClusteredForwardMaxGpuCullInstances),
                    // DX12 currently consumes direct draws while ExecuteIndirect
                    // is guarded. Keep graphics shaders on their root-constant
                    // path.
                    0u,
                    geometryBinCount,
                    0u);
                frameData.cameraNearFar = glm::vec4(
                    scene.camera.nearPlane, scene.camera.farPlane, 0.0f, 0.0f);
                return frameData;
            });
    }

    void DX12Backend::transitionResource(
        ID3D12GraphicsCommandList4* cmd,
        ID3D12Resource* resource,
        D3D12_RESOURCE_STATES before,
        D3D12_RESOURCE_STATES after)
    {
        if (!resource || before == after)
        {
            return;
        }

        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.Transition.pResource = resource;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = before;
        barrier.Transition.StateAfter = after;

        cmd->ResourceBarrier(1, &barrier);
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

    void DX12Backend::drawHiZDebugWindow()
    {
        if (!m_hiZDebugViewEnabled)
        {
            return;
        }

        ImGui::SetNextWindowSize(ImVec2(360.0f, 260.0f), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("Hi-Z Depth Pyramid", &m_hiZDebugViewEnabled))
        {
            ImGui::End();
            return;
        }

        const DX12GraphResourceEntry* hiZ =
            m_graphResourceRegistry.entry(
                m_clusteredForwardResources.hiZDebugResource);

        if (!hiZ)
        {
            ImGui::TextUnformatted("Waiting for Hi-Z pyramid...");
            ImGui::End();
            return;
        }

        const uint32_t mip =
            std::min<uint32_t>(m_hiZDebugMip, hiZ->mipLevels() - 1u);
        const uint32_t mipWidth = std::max(1u, hiZ->width() >> mip);
        const uint32_t mipHeight = std::max(1u, hiZ->height() >> mip);
        const ImVec2 available = ImGui::GetContentRegionAvail();
        const float maxWidth = std::max(1.0f, available.x);
        const float maxHeight = std::max(1.0f, available.y - 8.0f);
        const float aspect =
            static_cast<float>(mipWidth) / static_cast<float>(mipHeight);
        float width = maxWidth;
        float height = width / aspect;
        if (height > maxHeight)
        {
            height = maxHeight;
            width = height * aspect;
        }

        ImGui::Text("Mip %u / %u", mip, hiZ->mipLevels());
        ImGui::Text("%ux%u", mipWidth, mipHeight);
        ImGui::Image(
            (ImTextureID)hiZ->mipSrvs[mip].gpuStart.ptr,
            ImVec2(width, height));
        ImGui::End();
    }

    void DX12Backend::endDebugGuiFrame()
    {
        if (!m_imguiFrameActive)
        {
            return;
        }

        drawHiZDebugWindow();
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

    D3D12_RESOURCE_STATES DX12Backend::getOrInitResourceState(
        ID3D12Resource* resource)
    {
        auto it = m_resourceStates.find(resource);
        if (it != m_resourceStates.end())
        {
            return it->second.state;
        }

        m_resourceStates[resource] = {
            .state = D3D12_RESOURCE_STATE_PRESENT,
            .access = AccessType::Read
        };

        return D3D12_RESOURCE_STATE_PRESENT;
    }

    void DX12Backend::recreateSwapchain()
    {
        m_frameExecutor.waitForGpu();
        destroyDepthTarget();
        m_graphResourceRegistry.reset();
        destroyPathTraceResources();
        m_pipelineManager.shutdown();
        m_pipelineHandles.clear();
        m_computePipelineHandles.clear();
        m_swapchain.resize();
        m_pipelineManager.init(m_device);
        m_resourceStates.clear();
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
