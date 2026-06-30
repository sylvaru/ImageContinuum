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
#include "ic/renderer/dx12_backend/dx12_pipeline_manager.h"
#include "ic/renderer/dx12_backend/dx12_resource_allocator.h"
#include "ic/renderer/renderer_gpu_assets.h"

#include <d3d12.h>
#include <unordered_map>


namespace ic
{
    class Window;
    class PipelineLibrary;

	class DX12Backend : public RendererBackend
	{
	public:
        void initialize(
            const RendererSpecification& spec,
            const PipelineLibrary& pipelineLibrary,
            Window& window,
            uint32_t workerCount) override;

        void shutdown() override;

        void execute(
            const CompiledGraphPlan& plan,
            const FrameContext& ctx,
            const SceneRenderView& scene) override;

        DX12ResourceAllocator& resourceAllocator()
        {
            return m_resourceAllocator;
        }

        const DX12ResourceAllocator& resourceAllocator() const
        {
            return m_resourceAllocator;
        }

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

        struct UploadedModel
        {
            DX12Buffer vertexBuffer;
            DX12Buffer indexBuffer;
            std::vector<GpuMesh> meshes;
            std::vector<glm::mat4> meshTransforms;
            std::vector<GpuMaterialData> materials;
            bool uploaded = false;
        };

        struct FrameSceneResources
        {
            DX12Buffer frameConstants;
            DX12Buffer objects;
            DX12Buffer materials;

            DX12DescriptorAllocation objectSrv;
            DX12DescriptorAllocation materialSrv;

            uint32_t objectCapacity = 0;
            uint32_t materialCapacity = 0;
        };

        struct DrawItem
        {
            UploadedModel* model = nullptr;
            uint32_t objectIndex = 0;
            uint32_t meshIndex = 0;
            uint32_t materialIndex = 0;
        };

        void executeGraph(
            const CompiledGraphPlan& plan,
            const FrameContext& ctx,
            const SceneRenderView& scene,
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
            const CompiledGraphPlan& plan,
            const ExecutionNode& node,
            const FrameContext& ctx,
            const SceneRenderView& scene,
            ID3D12GraphicsCommandList4* cmd,
            ID3D12Resource* swapchainImage);

        void executeGraphicsNode(
            const CompiledGraphPlan& plan,
            const ExecutionNode& node,
            const FrameContext& ctx,
            const SceneRenderView& scene,
            ID3D12GraphicsCommandList4* cmd,
            ID3D12Resource* swapchainImage);

        void executeComputeNode(
            const CompiledGraphPlan& plan,
            const ExecutionNode& node,
            const FrameContext& ctx,
            ID3D12GraphicsCommandList4* cmd);

        void executeTransferNode(
            const CompiledGraphPlan& plan,
            const ExecutionNode& node,
            const FrameContext& ctx,
            ID3D12GraphicsCommandList4* cmd);

        void destroySceneResources();
        void ensureDepthTarget();
        void destroyDepthTarget();
        void ensureComputeTestBuffer();

        UploadedModel* requestModel(
            AssetHandle handle,
            const AssetManager& assets);

        bool prepareSceneResources(
            const FrameContext& ctx,
            const SceneRenderView& scene,
            std::vector<DrawItem>& draws);

        GraphicsPipelineHandle pipelineForNode(
            const CompiledGraphPlan& plan,
            const ExecutionNode& node);

        ComputePipelineHandle computePipelineForNode(
            const CompiledGraphPlan& plan,
            const ExecutionNode& node);

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
        TextureFormat swapchainTextureFormat() const;

        DX12Factory m_factory;
        DX12Adapter m_adapter;
        DX12Device m_device;
        DX12Swapchain m_swapchain;
        DX12CommandSystem m_commandSystem;
        DX12DescriptorSystem m_descriptorSystem;
        DX12PipelineManager m_pipelineManager;
        DX12ResourceAllocator m_resourceAllocator;

        DX12Texture m_depthTexture;
        DX12DescriptorAllocation m_depthDsv;
        D3D12_RESOURCE_STATES m_depthState = D3D12_RESOURCE_STATE_COMMON;
        uint32_t m_depthWidth = 0;
        uint32_t m_depthHeight = 0;
        DX12Buffer m_computeTestBuffer;
        D3D12_RESOURCE_STATES m_computeTestBufferState =
            D3D12_RESOURCE_STATE_COMMON;

        const PipelineLibrary* m_pipelineLibrary = nullptr;
        std::unordered_map<PipelineId, GraphicsPipelineHandle, PipelineIdHash> m_pipelineHandles;
        std::unordered_map<PipelineId, ComputePipelineHandle, PipelineIdHash> m_computePipelineHandles;
        std::unordered_map<AssetHandle, UploadedModel, AssetHandleHash> m_uploadedModels;
        std::vector<FrameSceneResources> m_sceneFrameResources;

        Microsoft::WRL::ComPtr<ID3D12Fence> m_fence;
        HANDLE m_fenceEvent = nullptr;
        uint64_t m_nextFenceValue = 1;
        uint32_t m_workerSlots = 1;
        std::vector<FrameSync> m_frameSync;
        std::unordered_map<ID3D12Resource*, ResourceState> m_resourceStates;
	};
}
