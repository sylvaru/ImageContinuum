#include "ic/common/ic_pch.h"
#include "ic/renderer/dx12_backend/dx12_backend.h"

#include "ic/core/frame_context.h"
#include "ic/core/app_base.h"
#include "ic/interface/window.h"
#include "ic/scene/scene_render_view.h"
#include "ic/renderer/pipeline_library.h"
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
                throw std::runtime_error(message);
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

        initFrameSync(spec);

        m_commandSystem.init(
            m_device.device(),
            framesInFlight,
            workerSlots);

        m_descriptorSystem.init(m_device);
        m_pipelineManager.init(m_device);
        m_sceneFrameResources.resize(framesInFlight);
        m_pipelineLibrary = &pipelineLibrary;

        if (spec.useDebugGui)
        {
            initImGui(window);
        }

        spdlog::info("[DX12Backend] Initialized");
    }

    void DX12Backend::shutdown()
    {
        waitForGpu();

        shutdownImGui();
        destroySceneResources();
        destroyEnvironmentResources();
        destroyClusteredForwardResources();
        destroyPathTraceResources();
        destroyGraphResources();
        destroyFrameSync();
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
        if (m_frameSync.empty())
        {
            return;
        }

        if (planUsesPathTracing(plan))
        {
            waitForGpu();
        }

        const uint32_t frameSlot =
            static_cast<uint32_t>(ctx.frameIndex % m_frameSync.size());

        waitForFrame(frameSlot);

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

        materializeGraphResources(plan, swapchainImage);

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

        if (!commandLists.empty())
        {
            m_device.graphicsQueue()->ExecuteCommandLists(
                static_cast<UINT>(commandLists.size()),
                commandLists.data());
        }

        const bool presented = m_swapchain.present();
        signalFrame(frameSlot);

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
        auto recordNode =
            [&](GraphNodeId nodeId, uint32_t workerIndex)
            {
                auto lease =
                    m_commandSystem.acquireFrameCommandList(
                        static_cast<uint32_t>(ctx.frameIndex % m_frameSync.size()),
                        workerIndex);

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

                throwIfFailed(
                    cmd->Close(),
                    "Failed to close DX12 frame command list.");

                return static_cast<ID3D12CommandList*>(cmd);
            };

        if (plan.executionLevels.empty())
        {
            for (uint32_t i = 0; i < plan.executionOrder.size(); ++i)
            {
                commandLists.push_back(
                    recordNode(plan.executionOrder[i], i % m_workerSlots));
            }

            return;
        }

        for (const ExecutionLevel& level : plan.executionLevels)
        {
            for (uint32_t i = 0; i < level.nodeCount; ++i)
            {
                const GraphNodeId nodeId =
                    plan.executionLevelNodes[level.firstNode + i];

                commandLists.push_back(
                    recordNode(nodeId, i % m_workerSlots));
            }
        }
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

            recordBarrier(
                cmd,
                plan.barriers[barrierIndex],
                plan.resources,
                swapchainImage);
        }
    }

    void DX12Backend::recordBarrier(
        ID3D12GraphicsCommandList4* cmd,
        const ResourceBarrier& barrier,
        std::span<const GraphResource> resources,
        ID3D12Resource* swapchainImage)
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
        else if (GraphResourceEntry* entry = graphResource(barrier.resource))
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
            resource.ownership == ResourceOwnership::Transient &&
                graphResource(barrier.resource)
                ? graphResource(barrier.resource)->state
                : usageToState(barrier.oldUsage);

        const D3D12_RESOURCE_STATES after =
            usageToState(barrier.newUsage);

        if (before == after)
        {
            return;
        }

        transitionResource(
            cmd,
            dxResource,
            before,
            after);

        if (resource.ownership == ResourceOwnership::Transient)
        {
            if (GraphResourceEntry* entry = graphResource(barrier.resource))
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
            findGraphAttachment(plan, node.nodeId, ResourceUsage::ColorAttachment);
        const GraphResourceId depthResource =
            findGraphAttachment(plan, node.nodeId, ResourceUsage::DepthAttachment);
        const GraphResourceEntry* colorEntry =
            colorResource != InvalidGraphResourceId
                ? graphResource(colorResource)
                : nullptr;
        const GraphResourceEntry* depthEntry =
            depthResource != InvalidGraphResourceId
                ? graphResource(depthResource)
                : nullptr;
        constexpr bool useGraphAttachments = false;

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
            m_preparedScene.draws.empty())
        {
            return;
        }

        const std::vector<DrawItem>& draws = m_preparedScene.draws;

        const uint32_t frameSlot =
            static_cast<uint32_t>(ctx.frameIndex % m_sceneFrameResources.size());
        FrameSceneResources& frameResources = m_sceneFrameResources[frameSlot];

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
        }
        cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

        UploadedModel* boundModel = nullptr;
        for (const DrawItem& draw : draws)
        {
            if (!draw.model ||
                draw.meshIndex >= draw.model->meshes.size())
            {
                continue;
            }

            const GpuMesh& mesh = draw.model->meshes[draw.meshIndex];
            if (mesh.indexCount == 0)
            {
                continue;
            }

            if (boundModel != draw.model)
            {
                D3D12_VERTEX_BUFFER_VIEW vbv{};
                vbv.BufferLocation = draw.model->vertexBuffer.gpuAddress;
                vbv.SizeInBytes =
                    static_cast<UINT>(draw.model->vertexBuffer.size);
                vbv.StrideInBytes = sizeof(AssetVertex);

                D3D12_INDEX_BUFFER_VIEW ibv{};
                ibv.BufferLocation = draw.model->indexBuffer.gpuAddress;
                ibv.SizeInBytes =
                    static_cast<UINT>(draw.model->indexBuffer.size);
                ibv.Format = DXGI_FORMAT_R32_UINT;

                cmd->IASetVertexBuffers(0, 1, &vbv);
                cmd->IASetIndexBuffer(&ibv);
                boundModel = draw.model;
            }

            DrawConstants constants{};
            constants.objectIndex = draw.objectIndex;
            constants.meshIndex = draw.meshIndex;
            constants.materialIndex = draw.materialIndex;

            const UINT drawRootParameter = 3u;
            cmd->SetGraphicsRoot32BitConstants(
                drawRootParameter,
                4,
                &constants,
                0);

            cmd->DrawIndexedInstanced(
                mesh.indexCount,
                1,
                mesh.firstIndex,
                0,
                0);
        }
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
            PipelineBindingLayoutKind::ClusteredForward)
        {
            if (!bindClusteredForwardCompute(*pipeline, ctx, scene, cmd))
            {
                return;
            }
        }

        cmd->Dispatch(
            pass->groupCountX,
            pass->groupCountY,
            pass->groupCountZ);

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
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        m_environmentResources.cubemapState =
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
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

        for (FrameSceneResources& resources : m_sceneFrameResources)
        {
            m_resourceAllocator.destroyBuffer(resources.frameConstants);
            m_resourceAllocator.destroyBuffer(resources.objects);
            m_resourceAllocator.destroyBuffer(resources.materials);
            m_resourceAllocator.destroyBuffer(resources.visibleLights);
            m_descriptorSystem.releaseResourceDescriptors(resources.objectSrv);
            m_descriptorSystem.releaseResourceDescriptors(resources.materialSrv);
            resources = {};
        }
        m_sceneFrameResources.clear();
        m_preparedScene = {};

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
            static_cast<uint32_t>(std::max<size_t>(1, m_frameSync.size()));
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
            waitForGpu();
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
            waitForGpu();

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

        if (m_clusteredForwardResources.clusterBounds &&
            m_clusteredForwardResources.width == width &&
            m_clusteredForwardResources.height == height)
        {
            return;
        }

        destroyClusteredForwardResources();

        m_clusteredForwardResources.width = width;
        m_clusteredForwardResources.height = height;
        m_clusteredForwardResources.clusterCountX = clusterCountX;
        m_clusteredForwardResources.clusterCountY = clusterCountY;
        m_clusteredForwardResources.clusterCountZ = clusterCountZ;
        m_clusteredForwardResources.clusterCount = clusterCount;

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

        m_clusteredForwardResources.boundsState =
            m_clusteredForwardResources.clusterBounds.initialState;
        m_clusteredForwardResources.gridState =
            m_clusteredForwardResources.clusterLightGrid.initialState;
        m_clusteredForwardResources.indicesState =
            m_clusteredForwardResources.clusterLightIndices.initialState;
        m_clusteredForwardResources.counterState =
            m_clusteredForwardResources.clusterLightCounter.initialState;

        spdlog::info(
            "[DX12Backend] Clustered forward resources: {}x{}x{} clusters",
            clusterCountX,
            clusterCountY,
            clusterCountZ);
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
            static_cast<uint32_t>(ctx.frameIndex % m_sceneFrameResources.size());
        FrameSceneResources& frameResources = m_sceneFrameResources[frameSlot];

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
            static_cast<uint32_t>(ctx.frameIndex % m_sceneFrameResources.size());
        FrameSceneResources& frameResources = m_sceneFrameResources[frameSlot];

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
                .usage = TextureUsageFlags::DepthAttachment,
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

        m_depthState = m_depthTexture.initialState;
        m_depthWidth = width;
        m_depthHeight = height;
    }

    void DX12Backend::destroyDepthTarget()
    {
        m_resourceAllocator.destroyTexture(m_depthTexture);
        m_descriptorSystem.releaseDSV(m_depthDsv);
        m_depthDsv = {};
        m_depthState = D3D12_RESOURCE_STATE_COMMON;
        m_depthWidth = 0;
        m_depthHeight = 0;
    }

    DX12Backend::GraphResourceEntry* DX12Backend::graphResource(
        GraphResourceId id)
    {
        auto it = m_graphResources.find(id);
        return it != m_graphResources.end() ? &it->second : nullptr;
    }

    const DX12Backend::GraphResourceEntry* DX12Backend::graphResource(
        GraphResourceId id) const
    {
        auto it = m_graphResources.find(id);
        return it != m_graphResources.end() ? &it->second : nullptr;
    }

    GraphResourceId DX12Backend::findGraphAttachment(
        const CompiledGraphPlan& plan,
        GraphNodeId node,
        ResourceUsage usage) const
    {
        for (const ResourceBarrier& barrier : plan.barriers)
        {
            if (barrier.toNode == node && barrier.newUsage == usage)
            {
                return barrier.resource;
            }
        }
        return InvalidGraphResourceId;
    }

    void DX12Backend::destroyGraphResources()
    {
        for (auto& [id, entry] : m_graphResources)
        {
            m_resourceAllocator.destroyTexture(entry.texture);
            m_resourceAllocator.destroyBuffer(entry.buffer);
            m_descriptorSystem.releaseRTV(entry.rtv);
            m_descriptorSystem.releaseDSV(entry.dsv);
            m_descriptorSystem.releaseResourceDescriptors(entry.srvUav);
        }
        m_graphResources.clear();
    }

    void DX12Backend::materializeGraphResources(
        const CompiledGraphPlan& plan,
        [[maybe_unused]] ID3D12Resource* swapchainImage)
    {
        const uint32_t extentWidth = m_swapchain.width();
        const uint32_t extentHeight = m_swapchain.height();
        for (const GraphResource& resource : plan.resources)
        {
            GraphResourceEntry& entry = m_graphResources[resource.id];
            entry.type = resource.type;
            entry.ownership = resource.ownership;
            entry.imported = resource.imported;

            if (resource.ownership == ResourceOwnership::Imported)
            {
                entry.width = extentWidth;
                entry.height = extentHeight;
                continue;
            }

            if (resource.type == GraphResourceType::Texture)
            {
                TextureDesc desc = resource.textureDesc;
                if (desc.width == 0) desc.width = extentWidth;
                if (desc.height == 0) desc.height = extentHeight;

                const bool recreate =
                    !entry.texture ||
                    entry.width != desc.width ||
                    entry.height != desc.height ||
                    entry.mipLevels != desc.mipLevels ||
                    entry.arrayLayers != desc.arrayLayers;
                if (!recreate)
                {
                    continue;
                }

                m_resourceAllocator.destroyTexture(entry.texture);
                m_descriptorSystem.releaseRTV(entry.rtv);
                m_descriptorSystem.releaseDSV(entry.dsv);
                m_descriptorSystem.releaseResourceDescriptors(entry.srvUav);
                entry.rtv = {};
                entry.dsv = {};
                entry.srvUav = {};

                entry.texture = m_resourceAllocator.createTexture(desc);
                entry.state = entry.texture.initialState;
                entry.width = desc.width;
                entry.height = desc.height;
                entry.mipLevels = desc.mipLevels;
                entry.arrayLayers = desc.arrayLayers;

                if (hasFlag(desc.usage, TextureUsageFlags::ColorAttachment))
                {
                    entry.rtv = m_descriptorSystem.allocateRTV(1);
                    D3D12_RENDER_TARGET_VIEW_DESC rtv{};
                    rtv.Format = entry.texture.desc.Format;
                    rtv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
                    m_device.device()->CreateRenderTargetView(
                        entry.texture.resource.Get(),
                        &rtv,
                        entry.rtv.cpuStart);
                }
                if (hasFlag(desc.usage, TextureUsageFlags::DepthAttachment))
                {
                    entry.dsv = m_descriptorSystem.allocateDSV(1);
                    D3D12_DEPTH_STENCIL_VIEW_DESC dsv{};
                    dsv.Format = DXGI_FORMAT_D32_FLOAT;
                    dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
                    m_device.device()->CreateDepthStencilView(
                        entry.texture.resource.Get(),
                        &dsv,
                        entry.dsv.cpuStart);
                }
            }
            else
            {
                BufferDesc desc = resource.bufferDesc;

                const bool missingBuffer = !entry.buffer;
                const bool sizeMismatch =
                    entry.buffer && entry.buffer.size != desc.size;
                const bool usageMismatch =
                    entry.buffer && entry.buffer.usage != desc.usage;
                const bool memoryMismatch =
                    entry.buffer && entry.buffer.memoryUsage != desc.memoryUsage;
                const bool mappedMismatch =
                    entry.buffer && entry.buffer.mappedAtCreation != desc.mappedAtCreation;

                const bool recreate =
                    missingBuffer ||
                    sizeMismatch ||
                    usageMismatch ||
                    memoryMismatch ||
                    mappedMismatch;

                if (!recreate)
                {
                    continue;
                }

                waitForGpu();

                m_resourceAllocator.destroyBuffer(entry.buffer);

                entry.buffer = m_resourceAllocator.createBuffer(desc);
                entry.state = entry.buffer.initialState;
                spdlog::warn(
                    "[DX12Backend] Stored graph buffer '{}' id={} valid={} size={} usage={} memory={} mapped={}",
                    desc.debugName,
                    static_cast<uint32_t>(resource.id),
                    static_cast<bool>(entry.buffer),
                    entry.buffer.size,
                    static_cast<uint32_t>(entry.buffer.usage),
                    static_cast<uint32_t>(entry.buffer.memoryUsage),
                    entry.buffer.mappedAtCreation);
            }
        }
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
                waitForGpu();

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
        waitForGpu();

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

    DX12Backend::UploadedModel* DX12Backend::requestModel(
        AssetHandle handle,
        const AssetManager& assets)
    {
        if (!handle)
        {
            return nullptr;
        }

        auto [it, inserted] =
            m_uploadedModels.try_emplace(handle);
        UploadedModel& uploaded = it->second;
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
        waitForGpu();

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
        if (m_preparedScene.frameIndex == ctx.frameIndex &&
            m_preparedScene.valid)
        {
            return !m_sceneFrameResources.empty();
        }

        m_preparedScene.frameIndex = ctx.frameIndex;
        m_preparedScene.valid = false;
        m_preparedScene.draws.clear();
        m_preparedScene.objects.clear();
        m_preparedScene.materials.clear();
        m_preparedScene.materialOffsets.clear();

        if (!ctx.services ||
            !ctx.services->assetManager ||
            scene.camera.valid == 0 ||
            m_sceneFrameResources.empty())
        {
            return false;
        }

        AssetManager& assets = *ctx.services->assetManager;
        std::vector<DrawItem>& draws = m_preparedScene.draws;
        std::vector<GpuObjectData>& objects = m_preparedScene.objects;
        std::vector<GpuMaterialData>& materials = m_preparedScene.materials;
        std::vector<GpuVisibleLight> visibleLights;
        auto& materialOffsets = m_preparedScene.materialOffsets;

        m_uploadedModels.reserve(
            m_uploadedModels.size() + scene.models.size());
        objects.reserve(scene.models.size());
        materialOffsets.reserve(scene.models.size());

        for (const SceneModelRenderItem& item : scene.models)
        {
            UploadedModel* model = requestModel(item.model, assets);
            if (!model)
            {
                continue;
            }

            uint32_t materialOffset = 0;
            if (auto it = materialOffsets.find(item.model);
                it != materialOffsets.end())
            {
                materialOffset = it->second;
            }
            else
            {
                materialOffset = static_cast<uint32_t>(materials.size());
                materialOffsets.emplace(item.model, materialOffset);
                materials.insert(
                    materials.end(),
                    model->materials.begin(),
                    model->materials.end());
            }

            for (uint32_t meshIndex = 0;
                 meshIndex < model->meshes.size();
                 ++meshIndex)
            {
                const GpuMesh& mesh = model->meshes[meshIndex];
                const glm::mat4 meshWorld =
                    meshIndex < model->meshTransforms.size()
                        ? item.world * model->meshTransforms[meshIndex]
                        : item.world;

                const uint32_t objectIndex =
                    static_cast<uint32_t>(objects.size());

                GpuObjectData object{};
                object.world = meshWorld;
                object.inverseTransposeWorld =
                    glm::inverseTranspose(meshWorld);
                objects.push_back(object);

                DrawItem draw{};
                draw.model = model;
                draw.objectIndex = objectIndex;
                draw.meshIndex = meshIndex;
                draw.materialIndex = materialOffset + mesh.materialIndex;
                draws.push_back(draw);
            }
        }

        if (draws.empty())
        {
            return false;
        }

        if (materials.empty())
        {
            materials.push_back(GpuMaterialData{});
        }

        visibleLights.reserve(
            std::min<size_t>(
                scene.lights.size(),
                ClusteredForwardMaxVisibleLights));
        for (const SceneLightRenderItem& light : scene.lights)
        {
            if (light.type != LightType::Point ||
                visibleLights.size() >= ClusteredForwardMaxVisibleLights)
            {
                continue;
            }

            GpuVisibleLight gpuLight{};
            gpuLight.positionRange = glm::vec4(light.position, light.range);
            gpuLight.colorIntensity = glm::vec4(light.color, light.intensity);
            visibleLights.push_back(gpuLight);
        }
        if (visibleLights.empty())
        {
            visibleLights.push_back(GpuVisibleLight{});
        }

        const uint32_t frameSlot =
            static_cast<uint32_t>(ctx.frameIndex % m_sceneFrameResources.size());
        FrameSceneResources& frameResources = m_sceneFrameResources[frameSlot];

        constexpr uint64_t constantBufferAlignment = 256;
        const uint64_t frameSize =
            (sizeof(GpuFrameData) + constantBufferAlignment - 1u) &
            ~(constantBufferAlignment - 1u);

        if (!frameResources.frameConstants)
        {
            frameResources.frameConstants =
                m_resourceAllocator.createBuffer({
                    .size = frameSize,
                    .usage = BufferUsageFlags::Constant,
                    .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                    .mappedAtCreation = true,
                    .debugName = "DX12 frame constants"
                });
        }

        if (objects.size() > frameResources.objectCapacity)
        {
            m_resourceAllocator.destroyBuffer(frameResources.objects);
            m_descriptorSystem.releaseResourceDescriptors(
                frameResources.objectSrv);
            frameResources.objectSrv = {};

            frameResources.objects =
                m_resourceAllocator.createBuffer({
                    .size = objects.size() * sizeof(GpuObjectData),
                    .usage = BufferUsageFlags::None,
                    .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                    .mappedAtCreation = true,
                    .debugName = "DX12 object data"
                });
            frameResources.objectCapacity =
                static_cast<uint32_t>(objects.size());

            frameResources.objectSrv =
                m_descriptorSystem.allocateResourceDescriptors(1);

            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.Shader4ComponentMapping =
                D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            srv.Format = DXGI_FORMAT_UNKNOWN;
            srv.Buffer.NumElements = frameResources.objectCapacity;
            srv.Buffer.StructureByteStride = sizeof(GpuObjectData);

            m_device.device()->CreateShaderResourceView(
                frameResources.objects.resource.Get(),
                &srv,
                frameResources.objectSrv.cpuStart);
        }

        if (materials.size() > frameResources.materialCapacity)
        {
            m_resourceAllocator.destroyBuffer(frameResources.materials);
            m_descriptorSystem.releaseResourceDescriptors(
                frameResources.materialSrv);
            frameResources.materialSrv = {};

            frameResources.materials =
                m_resourceAllocator.createBuffer({
                    .size = materials.size() * sizeof(GpuMaterialData),
                    .usage = BufferUsageFlags::None,
                    .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                    .mappedAtCreation = true,
                    .debugName = "DX12 material data"
                });
            frameResources.materialCapacity =
                static_cast<uint32_t>(materials.size());

            frameResources.materialSrv =
                m_descriptorSystem.allocateResourceDescriptors(1);

            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.Shader4ComponentMapping =
                D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            srv.Format = DXGI_FORMAT_UNKNOWN;
            srv.Buffer.NumElements = frameResources.materialCapacity;
            srv.Buffer.StructureByteStride = sizeof(GpuMaterialData);

            m_device.device()->CreateShaderResourceView(
                frameResources.materials.resource.Get(),
                &srv,
                frameResources.materialSrv.cpuStart);
        }

        if (visibleLights.size() > frameResources.visibleLightCapacity)
        {
            m_resourceAllocator.destroyBuffer(frameResources.visibleLights);
            frameResources.visibleLights =
                m_resourceAllocator.createBuffer({
                    .size = visibleLights.size() * sizeof(GpuVisibleLight),
                    .usage = BufferUsageFlags::None,
                    .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                    .mappedAtCreation = true,
                    .debugName = "DX12 clustered visible lights"
                });
            frameResources.visibleLightCapacity =
                static_cast<uint32_t>(visibleLights.size());
        }

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
            static_cast<uint32_t>(
                std::min<size_t>(
                    visibleLights.size(),
                    ClusteredForwardMaxVisibleLights));
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

        std::memcpy(
            frameResources.frameConstants.mapped,
            &frameData,
            sizeof(frameData));
        std::memcpy(
            frameResources.objects.mapped,
            objects.data(),
            objects.size() * sizeof(GpuObjectData));
        std::memcpy(
            frameResources.materials.mapped,
            materials.data(),
            materials.size() * sizeof(GpuMaterialData));
        std::memcpy(
            frameResources.visibleLights.mapped,
            visibleLights.data(),
            visibleLights.size() * sizeof(GpuVisibleLight));

        m_preparedScene.valid = true;
        return true;
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
            static_cast<int>(std::max<size_t>(1, m_frameSync.size()));
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
                static_cast<uint32_t>(ctx.frameIndex % m_frameSync.size()),
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

    void DX12Backend::initFrameSync(const RendererSpecification& spec)
    {
        const uint32_t framesInFlight =
            spec.framesInFlight == 0 ? 1 : spec.framesInFlight;

        m_frameSync.resize(framesInFlight);

        throwIfFailed(
            m_device.device()->CreateFence(
                0,
                D3D12_FENCE_FLAG_NONE,
                IID_PPV_ARGS(&m_fence)),
            "Failed to create DX12 frame fence.");

        m_fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!m_fenceEvent)
        {
            throw std::runtime_error("Failed to create DX12 fence event.");
        }

        m_nextFenceValue = 1;
    }

    void DX12Backend::destroyFrameSync()
    {
        if (m_fenceEvent)
        {
            CloseHandle(m_fenceEvent);
            m_fenceEvent = nullptr;
        }

        m_fence.Reset();
        m_frameSync.clear();
        m_nextFenceValue = 1;
    }

    void DX12Backend::waitForFrame(uint32_t frameSlot)
    {
        if (!m_fence || frameSlot >= m_frameSync.size())
        {
            return;
        }

        const uint64_t fenceValue =
            m_frameSync[frameSlot].fenceValue;

        if (fenceValue == 0 || m_fence->GetCompletedValue() >= fenceValue)
        {
            return;
        }

        throwIfFailed(
            m_fence->SetEventOnCompletion(fenceValue, m_fenceEvent),
            "Failed to wait on DX12 frame fence.");

        WaitForSingleObject(m_fenceEvent, INFINITE);
    }

    void DX12Backend::signalFrame(uint32_t frameSlot)
    {
        const uint64_t fenceValue =
            m_nextFenceValue++;

        throwIfFailed(
            m_device.graphicsQueue()->Signal(m_fence.Get(), fenceValue),
            "Failed to signal DX12 frame fence.");

        m_frameSync[frameSlot].fenceValue = fenceValue;
    }

    void DX12Backend::waitForGpu()
    {
        if (!m_device.graphicsQueue() || !m_fence)
        {
            return;
        }

        const uint64_t fenceValue =
            m_nextFenceValue++;

        throwIfFailed(
            m_device.graphicsQueue()->Signal(m_fence.Get(), fenceValue),
            "Failed to signal DX12 idle fence.");

        if (m_fence->GetCompletedValue() < fenceValue)
        {
            throwIfFailed(
                m_fence->SetEventOnCompletion(fenceValue, m_fenceEvent),
                "Failed to wait for DX12 idle fence.");

            WaitForSingleObject(m_fenceEvent, INFINITE);
        }

        for (FrameSync& sync : m_frameSync)
        {
            sync.fenceValue = 0;
        }
    }

    void DX12Backend::recreateSwapchain()
    {
        waitForGpu();
        destroyDepthTarget();
        destroyGraphResources();
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
