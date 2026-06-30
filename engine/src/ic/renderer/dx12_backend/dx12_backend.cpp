#include "ic/renderer/dx12_backend/dx12_backend.h"

#include "ic/core/frame_context.h"
#include "ic/core/app_base.h"
#include "ic/interface/window.h"
#include "ic/scene/scene_render_view.h"
#include "ic/renderer/pipeline_library.h"
#include "ic/renderer/renderer_specification.h"

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <spdlog/spdlog.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_dx12.h>
#include <backends/imgui_impl_glfw.h>

#include <stdexcept>
#include <future>

namespace
{
    void throwIfFailed(HRESULT hr, const char* message)
    {
        if (FAILED(hr))
        {
            throw std::runtime_error(message);
        }
    }
}

namespace ic
{
	void DX12Backend::initialize(
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

        if (spec.enableImGui)
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

        m_imguiEnabled = false;
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
                    plan.barriers,
                    plan.resources,
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

        for (const std::pmr::vector<GraphNodeId>& level : plan.executionLevels)
        {
            std::vector<std::future<ID3D12CommandList*>> futures;
            futures.reserve(level.size());

            for (uint32_t i = 0; i < level.size(); ++i)
            {
                futures.push_back(
                    std::async(
                        std::launch::async,
                        recordNode,
                        level[i],
                        i % m_workerSlots));
            }

            for (auto& future : futures)
            {
                commandLists.push_back(future.get());
            }
        }
    }

    void DX12Backend::applyBarriers(
        ID3D12GraphicsCommandList4* cmd,
        std::span<const ResourceBarrier> barriers,
        std::span<const GraphResource> resources,
        const ExecutionNode& node,
        ID3D12Resource* swapchainImage)
    {
        for (const ResourceBarrier& barrier : barriers)
        {
            if (barrier.toNode != node.nodeId)
            {
                continue;
            }

            recordBarrier(
                cmd,
                barrier,
                resources,
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

        if (!dxResource)
        {
            return;
        }

        const D3D12_RESOURCE_STATES before =
            usageToState(barrier.oldUsage);

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
            executeComputeNode(plan, node, ctx, cmd);
            break;

        case GraphNodeType::Transfer:
            executeTransferNode(plan, node, ctx, cmd);
            break;

        case GraphNodeType::Present:
            break;
        }
    }

    void DX12Backend::executeGraphicsNode(
        const CompiledGraphPlan& plan,
        [[maybe_unused]] const ExecutionNode& node,
        const FrameContext& ctx,
        const SceneRenderView& scene,
        ID3D12GraphicsCommandList4* cmd,
        [[maybe_unused]] ID3D12Resource* swapchainImage)
    {
        ensureDepthTarget();

        DX12GraphicsPipeline* pipeline =
            m_pipelineManager.graphicsPipeline(
                pipelineForNode(plan, node));
        if (!pipeline)
        {
            return;
        }

        const bool hasColorTarget =
            pipeline->desc.colorAttachmentCount > 0;

        const D3D12_CPU_DESCRIPTOR_HANDLE rtv =
            m_swapchain.currentRtv();

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

        if (m_depthState != D3D12_RESOURCE_STATE_DEPTH_WRITE)
        {
            transitionResource(
                cmd,
                m_depthTexture.resource.Get(),
                m_depthState,
                D3D12_RESOURCE_STATE_DEPTH_WRITE);
            m_depthState = D3D12_RESOURCE_STATE_DEPTH_WRITE;
        }

        cmd->OMSetRenderTargets(
            hasColorTarget ? 1u : 0u,
            hasColorTarget ? &rtv : nullptr,
            FALSE,
            &m_depthDsv.cpuStart);

        constexpr FLOAT clearColor[4] = { 0.02f, 0.02f, 0.025f, 1.0f };
        if (hasColorTarget)
        {
            cmd->ClearRenderTargetView(rtv, clearColor, 0, nullptr);
        }
        else
        {
            cmd->ClearDepthStencilView(
                m_depthDsv.cpuStart,
                D3D12_CLEAR_FLAG_DEPTH,
                1.0f,
                0,
                0,
                nullptr);
        }

        std::vector<DrawItem> draws;
        if (!prepareSceneResources(ctx, scene, draws) || draws.empty())
        {
            return;
        }

        const uint32_t frameSlot =
            static_cast<uint32_t>(ctx.frameIndex % m_sceneFrameResources.size());
        FrameSceneResources& frameResources = m_sceneFrameResources[frameSlot];

        ID3D12DescriptorHeap* heaps[] =
        {
            m_descriptorSystem.shaderResourceHeap()
        };

        cmd->SetDescriptorHeaps(1, heaps);

        cmd->SetGraphicsRootSignature(pipeline->rootSignature.Get());
        cmd->SetPipelineState(pipeline->pipelineState.Get());
        cmd->SetGraphicsRootConstantBufferView(
            0,
            frameResources.frameConstants.gpuAddress);
        cmd->SetGraphicsRootDescriptorTable(
            1,
            frameResources.objectSrv.gpuStart);
        cmd->SetGraphicsRootDescriptorTable(
            2,
            frameResources.materialSrv.gpuStart);
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

            cmd->SetGraphicsRoot32BitConstants(
                3,
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
        [[maybe_unused]] ID3D12GraphicsCommandList4* cmd)
    {
        if (node.payloadIndex >= plan.payloads.size())
        {
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
    }

    void DX12Backend::executeTransferNode(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node,
        [[maybe_unused]] const FrameContext& ctx,
        [[maybe_unused]] ID3D12GraphicsCommandList4* cmd)
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

        // Transfer payload operations land here; resource barriers already run.
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

        const ComputePassData* pass =
            std::get_if<ComputePassData>(&plan.payloads[node.payloadIndex]);
        if (!pass || !pass->pipeline)
        {
            return {};
        }

        auto it = m_computePipelineHandles.find(pass->pipeline);
        if (it != m_computePipelineHandles.end())
        {
            return it->second;
        }

        ComputePipelineDesc desc =
            m_pipelineLibrary->resolveCompute(
                pass->pipeline,
                RendererBackendType::DX12);

        ComputePipelineHandle handle =
            m_pipelineManager.requestComputePipeline(desc);
        m_computePipelineHandles.emplace(pass->pipeline, handle);
        return handle;
    }

    void DX12Backend::destroySceneResources()
    {
        destroyDepthTarget();
        m_resourceAllocator.destroyBuffer(m_computeTestBuffer);
        m_computeTestBufferState = D3D12_RESOURCE_STATE_COMMON;

        for (auto& [handle, model] : m_uploadedModels)
        {
            m_resourceAllocator.destroyBuffer(model.vertexBuffer);
            m_resourceAllocator.destroyBuffer(model.indexBuffer);
        }
        m_uploadedModels.clear();

        for (FrameSceneResources& resources : m_sceneFrameResources)
        {
            m_resourceAllocator.destroyBuffer(resources.frameConstants);
            m_resourceAllocator.destroyBuffer(resources.objects);
            m_resourceAllocator.destroyBuffer(resources.materials);
            resources = {};
        }
        m_sceneFrameResources.clear();

        m_pipelineHandles.clear();
        m_computePipelineHandles.clear();
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
        m_depthDsv = {};
        m_depthState = D3D12_RESOURCE_STATE_COMMON;
        m_depthWidth = 0;
        m_depthHeight = 0;
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

        uploaded.vertexBuffer =
            m_resourceAllocator.createBuffer({
                .size = model->vertices.size() * sizeof(AssetVertex),
                .usage = BufferUsageFlags::Vertex,
                .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                .mappedAtCreation = true,
                .debugName = "Cornell vertex buffer"
            });

        uploaded.indexBuffer =
            m_resourceAllocator.createBuffer({
                .size = model->indices.size() * sizeof(uint32_t),
                .usage = BufferUsageFlags::Index,
                .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                .mappedAtCreation = true,
                .debugName = "Cornell index buffer"
            });

        std::memcpy(
            uploaded.vertexBuffer.mapped,
            model->vertices.data(),
            uploaded.vertexBuffer.size);
        std::memcpy(
            uploaded.indexBuffer.mapped,
            model->indices.data(),
            uploaded.indexBuffer.size);

        uploaded.materials.reserve(std::max<size_t>(1, model->materials.size()));
        for (const MaterialAsset& src : model->materials)
        {
            GpuMaterialData dst{};
            dst.baseColorFactor = src.baseColorFactor;
            dst.metallicFactor = src.metallicFactor;
            dst.roughnessFactor = src.roughnessFactor;
            dst.alphaCutoff = src.alphaCutoff;
            dst.flags =
                (src.doubleSided ? 1u : 0u) |
                (src.alphaBlend ? 2u : 0u) |
                (src.alphaMask ? 4u : 0u);
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
        const SceneRenderView& scene,
        std::vector<DrawItem>& draws)
    {
        if (!ctx.services ||
            !ctx.services->assetManager ||
            scene.camera.valid == 0 ||
            m_sceneFrameResources.empty())
        {
            return false;
        }

        AssetManager& assets = *ctx.services->assetManager;
        std::vector<GpuObjectData> objects;
        std::vector<GpuMaterialData> materials;
        std::unordered_map<AssetHandle, uint32_t, AssetHandleHash> materialOffsets;

        objects.reserve(scene.models.size());

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
        m_pipelineManager.shutdown();
        m_pipelineHandles.clear();
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
