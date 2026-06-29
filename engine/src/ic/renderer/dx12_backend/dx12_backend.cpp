#include "ic/renderer/dx12_backend/dx12_backend.h"

#include "ic/core/frame_context.h"
#include "ic/interface/window.h"
#include "ic/renderer/renderer_specification.h"

#include <spdlog/spdlog.h>

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
        Window& window,
        uint32_t workerCount)
    {
        spdlog::info("[DX12Backend] Initializing...");

        m_factory.init(spec.enableValidation);
        m_adapter.init(m_factory);
        m_device.init(m_adapter, m_factory.validationEnabled());

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

        spdlog::info("[DX12Backend] Initialized");
    }

    void DX12Backend::shutdown()
    {
        waitForGpu();

        destroyFrameSync();
        m_descriptorSystem.shutdown();
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
        const FrameContext& ctx)
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
                    node,
                    ctx,
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
        const ExecutionNode& node,
        const FrameContext& ctx,
        ID3D12GraphicsCommandList4* cmd,
        ID3D12Resource* swapchainImage)
    {
        switch (node.type)
        {
        case GraphNodeType::Graphics:
            executeGraphicsNode(node, ctx, cmd, swapchainImage);
            break;

        case GraphNodeType::Compute:
        case GraphNodeType::Transfer:
        case GraphNodeType::Present:
            break;
        }
    }

    void DX12Backend::executeGraphicsNode(
        [[maybe_unused]] const ExecutionNode& node,
        [[maybe_unused]] const FrameContext& ctx,
        ID3D12GraphicsCommandList4* cmd,
        [[maybe_unused]] ID3D12Resource* swapchainImage)
    {
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
        cmd->OMSetRenderTargets(1, &rtv, FALSE, nullptr);

        constexpr FLOAT clearColor[4] = { 1.0f, 0.0f, 1.0f, 1.0f };
        cmd->ClearRenderTargetView(rtv, clearColor, 0, nullptr);
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
        m_swapchain.resize();
        m_resourceStates.clear();
    }
}
