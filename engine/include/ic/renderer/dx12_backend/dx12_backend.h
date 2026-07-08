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
#include "ic/renderer/path_tracing/path_tracer_types.h"

#include <d3d12.h>
#include <glm/glm.hpp>
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

        void execute(
            const CompiledGraphPlan& plan,
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
            std::vector<uint32_t> textureDescriptorIndices;
            std::vector<uint32_t> samplerDescriptorIndices;
            bool uploaded = false;
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

        struct GraphResourceEntry
        {
            GraphResourceType type = GraphResourceType::Texture;
            ResourceOwnership ownership = ResourceOwnership::Transient;
            ImportedResource imported = ImportedResource::None;
            DX12Texture texture;
            DX12Buffer buffer;
            DX12DescriptorAllocation rtv;
            DX12DescriptorAllocation dsv;
            DX12DescriptorAllocation srvUav;
            D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
            uint32_t width = 0;
            uint32_t height = 0;
            uint32_t mipLevels = 1;
            uint32_t arrayLayers = 1;
        };

        struct FrameSceneResources
        {
            DX12Buffer frameConstants;
            DX12Buffer objects;
            DX12Buffer materials;
            DX12Buffer visibleLights;

            DX12DescriptorAllocation objectSrv;
            DX12DescriptorAllocation materialSrv;

            uint32_t objectCapacity = 0;
            uint32_t materialCapacity = 0;
            uint32_t visibleLightCapacity = 0;
        };

        struct DrawItem
        {
            UploadedModel* model = nullptr;
            uint32_t objectIndex = 0;
            uint32_t meshIndex = 0;
            uint32_t materialIndex = 0;
        };

        struct PreparedSceneFrame
        {
            uint64_t frameIndex = UINT64_MAX;
            bool valid = false;

            std::vector<DrawItem> draws;
            std::vector<GpuObjectData> objects;
            std::vector<GpuMaterialData> materials;
            std::unordered_map<AssetHandle, uint32_t, AssetHandleHash>
                materialOffsets;
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
            uint32_t clusterCountX = 0;
            uint32_t clusterCountY = 0;
            uint32_t clusterCountZ = 0;
            uint32_t clusterCount = 0;
        };

        void executeGraph(
            const CompiledGraphPlan& plan,
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
        void materializeGraphResources(
            const CompiledGraphPlan& plan,
            ID3D12Resource* swapchainImage);
        void destroyGraphResources();
        GraphResourceEntry* graphResource(GraphResourceId id);
        const GraphResourceEntry* graphResource(GraphResourceId id) const;
        GraphResourceId findGraphAttachment(
            const CompiledGraphPlan& plan,
            GraphNodeId node,
            ResourceUsage usage) const;
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
        void destroyClusteredForwardResources();

        UploadedModel* requestModel(
            AssetHandle handle,
            const AssetManager& assets);

        uint32_t requestTexture(
            AssetHandle modelHandle,
            uint32_t imageIndex,
            const ImageAsset& image,
            TextureTransferFunction transfer);

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
        ClusteredForwardResources m_clusteredForwardResources;
        PathTraceResources m_pathTraceResources;
        EnvironmentResources m_environmentResources;

        const PipelineLibrary* m_pipelineLibrary = nullptr;
        std::unordered_map<PipelineId, GraphicsPipelineHandle, PipelineIdHash> m_pipelineHandles;
        std::unordered_map<PipelineId, ComputePipelineHandle, PipelineIdHash> m_computePipelineHandles;
        std::unordered_map<AssetHandle, UploadedModel, AssetHandleHash> m_uploadedModels;
        std::unordered_map<uint64_t, UploadedTexture> m_uploadedTextures;
        std::unordered_map<uint64_t, UploadedSampler> m_uploadedSamplers;
        std::unordered_map<GraphResourceId, GraphResourceEntry>
            m_graphResources;
        std::vector<FrameSceneResources> m_sceneFrameResources;
        PreparedSceneFrame m_preparedScene;

        Microsoft::WRL::ComPtr<ID3D12Fence> m_fence;
        HANDLE m_fenceEvent = nullptr;
        uint64_t m_nextFenceValue = 1;
        uint32_t m_workerSlots = 1;
        std::vector<FrameSync> m_frameSync;
        std::unordered_map<ID3D12Resource*, ResourceState> m_resourceStates;
        bool m_imguiEnabled = false;
        bool m_imguiFrameActive = false;
        bool m_clusteredForwardHeatmapEnabled = false;
        std::string m_imguiIniPath;
	};
}
