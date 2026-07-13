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
#include "ic/renderer/dx12_backend/dx12_graph_resource_registry.h"
#include "ic/renderer/dx12_backend/dx12_frame_executor.h"
#include "ic/renderer/dx12_backend/dx12_gpu_scene.h"
#include "ic/renderer/dx12_backend/dx12_retirement_queue.h"
#include "ic/renderer/dx12_backend/dx12_upload_scheduler.h"
#include "ic/renderer/renderer_gpu_assets.h"
#include "ic/renderer/path_tracing/path_tracer_types.h"

#include <d3d12.h>
#include <glm/glm.hpp>
#include <mutex>
#include <unordered_map>
#include <vector>


namespace ic
{
    class Window;
    class PipelineLibrary;

	class DX12Backend : public RendererBackend
	{
	public:
        void init(
            const RendererSpecification& spec,
            const PipelineLibrary& pipelineLibrary,
            Window& window,
            uint32_t workerCount) override;

        void shutdown() override;

        [[nodiscard]] bool execute(
            const CompiledGraphPlan& plan,
            const GraphExecutionContext& execution,
            const FrameContext& ctx,
            const SceneRenderView& scene) override;

        std::vector<IBLBakeResult> executeIBLBakeRequests(
            std::span<const IBLBakeRequest> requests,
            const FrameContext& ctx) override;

        bool beginDebugGuiFrame() override;
        void endDebugGuiFrame() override;
        bool vsyncEnabled() const override;
        void setVsyncEnabled(bool enabled) override;
        bool clusteredForwardHeatmapEnabled() const override;
        void setClusteredForwardHeatmapEnabled(bool enabled) override;
        bool hiZDebugViewEnabled() const override;
        void setHiZDebugViewEnabled(bool enabled) override;
        uint32_t hiZDebugMip() const override;
        void setHiZDebugMip(uint32_t mip) override;

        DX12ResourceAllocator& resourceAllocator()
        {
            return m_resourceAllocator;
        }

        const DX12ResourceAllocator& resourceAllocator() const
        {
            return m_resourceAllocator;
        }

    private:
        struct ResourceState
        {
            D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
            AccessType access = AccessType::Read;
        };

        struct UploadedTexture
        {
            DX12Texture texture;
            DX12DescriptorAllocation srv;
            D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
        };

        struct UploadedSampler
        {
            DX12DescriptorAllocation descriptor;
        };

        struct PathTraceResources
        {
            DX12Texture accumulation;
            DX12Texture tonemap;
            std::vector<DX12Buffer> pathTraceConstants;
            std::vector<DX12Buffer> tonemapConstants;
            DX12Buffer sceneVertices;
            DX12Buffer sceneMaterials;
            DX12Buffer sceneTriangles;
            DX12Buffer sceneBvhNodes;

            DX12DescriptorAllocation accumulationUav;
            DX12DescriptorAllocation accumulationSrv;
            DX12DescriptorAllocation tonemapUav;
            DX12DescriptorAllocation sceneSrvs;

            D3D12_RESOURCE_STATES accumulationState =
                D3D12_RESOURCE_STATE_COMMON;
            D3D12_RESOURCE_STATES tonemapState =
                D3D12_RESOURCE_STATE_COMMON;

            uint32_t width = 0;
            uint32_t height = 0;
            uint32_t sceneVertexCount = 0;
            uint32_t sceneMaterialCount = 0;
            uint32_t sceneTriangleCount = 0;
            uint32_t sceneBvhNodeCount = 0;
            uint32_t firstEmissiveTriangleIndex = UINT32_MAX;
            uint32_t accumulatedSampleCount = 0;
            uint64_t sceneVersion = UINT64_MAX;
            uint64_t environmentVersion = UINT64_MAX;
            uint64_t lastSceneBuildFrame = UINT64_MAX;
            bool sceneHadPendingModels = false;
            float tonemapExposure = 1.0f;
            glm::mat4 previousView = glm::mat4(1.0f);
            glm::mat4 previousProjection = glm::mat4(1.0f);
            bool resetAccumulation = true;
            bool hasPreviousCamera = false;
        };

        struct EnvironmentResources
        {
            AssetHandle source = {};
            DX12Texture cubemap;
            DX12Texture irradiance;
            DX12Texture prefiltered;
            DX12Texture brdfLut;
            DX12DescriptorAllocation cubemapSrv;
            DX12DescriptorAllocation cubemapUav;
            DX12DescriptorAllocation irradianceSrv;
            DX12DescriptorAllocation irradianceUav;
            DX12DescriptorAllocation prefilteredSrv;
            std::vector<DX12DescriptorAllocation> prefilteredUavs;
            DX12DescriptorAllocation brdfLutSrv;
            DX12DescriptorAllocation brdfLutUav;
            DX12DescriptorAllocation sampler;
            std::vector<DX12Buffer> skyboxConstants;
            D3D12_RESOURCE_STATES cubemapState = D3D12_RESOURCE_STATE_COMMON;
            D3D12_RESOURCE_STATES irradianceState = D3D12_RESOURCE_STATE_COMMON;
            D3D12_RESOURCE_STATES prefilteredState = D3D12_RESOURCE_STATE_COMMON;
            D3D12_RESOURCE_STATES brdfLutState = D3D12_RESOURCE_STATE_COMMON;
            uint32_t cubemapSize = 512;
            uint32_t irradianceSize = 64;
            uint32_t prefilterSize = 256;
            uint32_t prefilterMipCount = 1;
            uint32_t brdfLutSize = 512;
            bool converted = false;
            bool iblBaked = false;
        };

        // Clustered-light-grid resources. The GPU-driven cull/indirect-draw
        // buffers live in DX12GpuScene (m_gpuScene) instead, since they are a
        // separate concern (draw submission, not light clustering).
        struct ClusteredForwardResources
        {
            DX12Buffer clusterBounds;
            DX12Buffer clusterLightGrid;
            DX12Buffer clusterLightIndices;
            DX12Buffer clusterLightCounter;
            D3D12_RESOURCE_STATES boundsState = D3D12_RESOURCE_STATE_COMMON;
            D3D12_RESOURCE_STATES gridState = D3D12_RESOURCE_STATE_COMMON;
            D3D12_RESOURCE_STATES indicesState = D3D12_RESOURCE_STATE_COMMON;
            D3D12_RESOURCE_STATES counterState = D3D12_RESOURCE_STATE_COMMON;
            uint32_t width = 0;
            uint32_t height = 0;
            uint32_t hiZMipCount = 0;
            GraphResourceId hiZDebugResource = InvalidGraphResourceId;
            bool loggedHiZ = false;
            bool loggedHiZDebugResource = false;
            uint32_t clusterCountX = 0;
            uint32_t clusterCountY = 0;
            uint32_t clusterCountZ = 0;
            uint32_t clusterCount = 0;
        };

        void executeGraph(
            const CompiledGraphPlan& plan,
            const GraphExecutionContext& execution,
            const FrameContext& ctx,
            const SceneRenderView& scene,
            ID3D12Resource* swapchainImage,
            std::vector<ID3D12CommandList*>& commandLists);

        void initImGui(Window& window);
        void shutdownImGui();
        void recordImGui(
            const FrameContext& ctx,
            ID3D12Resource* swapchainImage,
            std::vector<ID3D12CommandList*>& commandLists);

        void applyBarriers(
            ID3D12GraphicsCommandList4* cmd,
            const CompiledGraphPlan& plan,
            const ExecutionNode& node,
            ID3D12Resource* swapchainImage);

        void recordBarrier(
            ID3D12GraphicsCommandList4* cmd,
            const ResourceBarrier& barrier,
            std::span<const GraphResource> resources,
            ID3D12Resource* swapchainImage,
            bool crossQueueRelease = false,
            bool crossQueueAcquire = false,
            QueueType commandQueue = QueueType::Graphics);

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
            const SceneRenderView& scene,
            ID3D12GraphicsCommandList4* cmd);

        void executePathTraceNode(
            const CompiledGraphPlan& plan,
            const ExecutionNode& node,
            const FrameContext& ctx,
            const SceneRenderView& scene,
            ID3D12GraphicsCommandList4* cmd);

        void executeEnvironmentConvertNode(
            const CompiledGraphPlan& plan,
            const ExecutionNode& node,
            const FrameContext& ctx,
            const SceneRenderView& scene,
            ID3D12GraphicsCommandList4* cmd);

        void executeTonemapNode(
            const CompiledGraphPlan& plan,
            const ExecutionNode& node,
            const FrameContext& ctx,
            ID3D12GraphicsCommandList4* cmd);

        void executeTransferNode(
            const CompiledGraphPlan& plan,
            const ExecutionNode& node,
            const FrameContext& ctx,
            ID3D12GraphicsCommandList4* cmd,
            ID3D12Resource* swapchainImage);

        void destroySceneResources();
        void ensurePathTraceResources();
        void ensurePathTraceSceneResources(
            const FrameContext& ctx,
            const SceneRenderView& scene);
        void destroyPathTraceResources();
        void destroyPathTraceSceneResources();
        void retirePathTraceSceneResources();
        void uploadPathTraceScene(const PathTraceSceneData& sceneData);
        void updatePathTraceDescriptors();
        bool ensureEnvironmentResources(
            const FrameContext& ctx,
            const SceneRenderView& scene,
            ID3D12GraphicsCommandList4* cmd);
        DX12ComputePipeline* environmentConvertPipeline();
        bool convertEnvironmentIfReady(
            DX12ComputePipeline& pipeline,
            const FrameContext& ctx,
            const SceneRenderView& scene,
            ID3D12GraphicsCommandList4* cmd);
        void destroyEnvironmentResources();
        void drawSkybox(
            DX12GraphicsPipeline& pipeline,
            const FrameContext& ctx,
            const SceneRenderView& scene,
            ID3D12GraphicsCommandList4* cmd);
        void ensureDepthTarget();
        void destroyDepthTarget();
        void drawHiZDebugWindow();
        void ensureComputeTestBuffer();
        void ensureClusteredForwardResources();
        bool bindClusteredForwardCompute(
            const DX12ComputePipeline& pipeline,
            const FrameContext& ctx,
            const SceneRenderView& scene,
            ID3D12GraphicsCommandList4* cmd);
        void bindClusteredForwardGraphics(
            const DX12GraphicsPipeline& pipeline,
            const FrameContext& ctx,
            ID3D12GraphicsCommandList4* cmd);
        void readbackVisibleInstanceCount(
            const FrameContext& ctx,
            ID3D12GraphicsCommandList4* cmd);
        void destroyClusteredForwardResources();

        DX12UploadedModel* requestModel(
            AssetHandle handle,
            const AssetManager& assets);

        uint32_t requestTexture(
            AssetHandle modelHandle,
            uint32_t imageIndex,
            const ImageAsset& image,
            TextureTransferFunction transfer,
            ID3D12GraphicsCommandList4* immediateCommandList = nullptr);

        uint32_t requestSampler(const SamplerAsset* sampler);

        bool prepareSceneResources(
            const FrameContext& ctx,
            const SceneRenderView& scene);

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
        DX12DescriptorAllocation m_depthSrv;
        D3D12_RESOURCE_STATES m_depthState = D3D12_RESOURCE_STATE_COMMON;
        uint32_t m_depthWidth = 0;
        uint32_t m_depthHeight = 0;
        DX12Buffer m_computeTestBuffer;
        D3D12_RESOURCE_STATES m_computeTestBufferState =
            D3D12_RESOURCE_STATE_COMMON;
        ClusteredForwardResources m_clusteredForwardResources;
        std::mutex m_clusteredForwardResourcesMutex;
        PathTraceResources m_pathTraceResources;
        EnvironmentResources m_environmentResources;

        const PipelineLibrary* m_pipelineLibrary = nullptr;
        std::unordered_map<PipelineId, GraphicsPipelineHandle, PipelineIdHash> m_pipelineHandles;
        std::unordered_map<PipelineId, ComputePipelineHandle, PipelineIdHash> m_computePipelineHandles;
        std::unordered_map<AssetHandle, DX12UploadedModel, AssetHandleHash> m_uploadedModels;
        std::unordered_map<uint64_t, UploadedTexture> m_uploadedTextures;
        std::unordered_map<uint64_t, UploadedSampler> m_uploadedSamplers;
        DX12GraphResourceRegistry m_graphResourceRegistry;
        DX12GpuScene m_gpuScene;
        DX12UploadScheduler m_uploadScheduler;
        DX12RetirementQueue m_retirementQueue;

        DX12FrameExecutor m_frameExecutor;
        uint32_t m_workerSlots = 1;
        std::unordered_map<ID3D12Resource*, ResourceState> m_resourceStates;
        bool m_imguiEnabled = false;
        bool m_imguiFrameActive = false;
        bool m_clusteredForwardHeatmapEnabled = false;
        bool m_hiZDebugViewEnabled = false;
        uint32_t m_hiZDebugMip = 0;
        std::string m_imguiIniPath;
	};
}
