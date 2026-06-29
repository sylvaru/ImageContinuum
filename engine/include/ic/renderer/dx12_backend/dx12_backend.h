#pragma once
#include "ic/renderer/renderer_backend.h"
#include "ic/renderer/frame_graph/compiled_graph_plan.h"
#include "ic/renderer/frame_graph/frame_graph_types.h"
#include "ic/renderer/dx12_backend/dx12_factory.h"
#include "ic/renderer/dx12_backend/dx12_adapter.h"
#include "ic/renderer/dx12_backend/dx12_device.h"
#include "ic/renderer/dx12_backend/dx12_swapchain.h"
#include "ic/renderer/dx12_backend/dx12_command_system.h"
#include "ic/renderer/dx12_backend/dx12_descriptor_system.h"

#include <d3d12.h>
#include <unordered_map>


namespace ic
{
    class Window;

	class DX12Backend : public RendererBackend
	{
	public:
        void initialize(
            const RendererSpecification& spec,
            Window& window,
            uint32_t workerCount) override;

        void shutdown() override;

        void execute(
            const CompiledGraphPlan& plan,
            const FrameContext& ctx) override;

    private:
        struct FrameSync
        {
            uint64_t fenceValue = 0;
        };

        struct ResourceState
        {
            D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
            AccessType access = AccessType::Read;
        };

        void executeGraph(
            const CompiledGraphPlan& plan,
            const FrameContext& ctx,
            ID3D12Resource* swapchainImage,
            std::vector<ID3D12CommandList*>& commandLists);

        void applyBarriers(
            ID3D12GraphicsCommandList4* cmd,
            std::span<const ResourceBarrier> barriers,
            std::span<const GraphResource> resources,
            const ExecutionNode& node,
            ID3D12Resource* swapchainImage);

        void recordBarrier(
            ID3D12GraphicsCommandList4* cmd,
            const ResourceBarrier& barrier,
            std::span<const GraphResource> resources,
            ID3D12Resource* swapchainImage);

        void dispatchNode(
            const ExecutionNode& node,
            const FrameContext& ctx,
            ID3D12GraphicsCommandList4* cmd,
            ID3D12Resource* swapchainImage);

        void executeGraphicsNode(
            const ExecutionNode& node,
            const FrameContext& ctx,
            ID3D12GraphicsCommandList4* cmd,
            ID3D12Resource* swapchainImage);

        void transitionResource(
            ID3D12GraphicsCommandList4* cmd,
            ID3D12Resource* resource,
            D3D12_RESOURCE_STATES before,
            D3D12_RESOURCE_STATES after);

        D3D12_RESOURCE_STATES usageToState(ResourceUsage usage) const;
        D3D12_RESOURCE_STATES getOrInitResourceState(ID3D12Resource* resource);

        void initFrameSync(const RendererSpecification& spec);
        void destroyFrameSync();
        void waitForFrame(uint32_t frameSlot);
        void signalFrame(uint32_t frameSlot);
        void waitForGpu();
        void recreateSwapchain();

        DX12Factory m_factory;
        DX12Adapter m_adapter;
        DX12Device m_device;
        DX12Swapchain m_swapchain;
        DX12CommandSystem m_commandSystem;
        DX12DescriptorSystem m_descriptorSystem;

        Microsoft::WRL::ComPtr<ID3D12Fence> m_fence;
        HANDLE m_fenceEvent = nullptr;
        uint64_t m_nextFenceValue = 1;
        uint32_t m_workerSlots = 1;
        std::vector<FrameSync> m_frameSync;
        std::unordered_map<ID3D12Resource*, ResourceState> m_resourceStates;
	};
}
