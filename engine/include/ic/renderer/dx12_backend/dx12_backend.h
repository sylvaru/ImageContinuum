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
#include "ic/renderer/dx12_backend/dx12_pass_recorders.h"
#include "ic/renderer/dx12_backend/dx12_frame_executor.h"
#include "ic/renderer/dx12_backend/dx12_gpu_scene.h"
#include "ic/renderer/dx12_backend/dx12_retirement_queue.h"
#include "ic/renderer/dx12_backend/dx12_upload_scheduler.h"
#include "ic/renderer/dx12_backend/dx12_gpu_profiler.h"
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

        [[nodiscard]] RenderSurfaceState reconcileRenderSurface() override;

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
        bool hiZDebugPrevious() const override;
        void setHiZDebugPrevious(bool previous) override;
        bool gpuOcclusionEnabled() const override;
        void setGpuOcclusionEnabled(bool enabled) override;
        GpuCullDebugMode gpuCullDebugMode() const override;
        void setGpuCullDebugMode(GpuCullDebugMode mode) override;
        GpuCullStats gpuCullStats() const override;
        GpuCullPerformance gpuCullPerformance() const override;
        // D3D12 exposes a dedicated compute engine and the frame executor
        // submits per-queue batches to m_device.computeQueue() with per-queue
        // fence waits, so the clustered-forward light chain runs truly async
        // (validated: overlaps the depth prepass on separate queues, stable over
        // repeated frame-0 and sustained runs, zero device removals).
        //
        // Cross-queue correctness for the frame-graph-owned cluster buffers
        // (bound through root descriptors) relies on D3D12 implicit buffer
        // promotion/decay rather than explicit transition barriers: buffers
        // promote COMMON -> UAV/SRV on access and decay back to COMMON after
        // every ExecuteCommandLists, so they are always COMMON at a queue
        // boundary and no before-state is ever asserted across queues (see
        // recordBarrier's buffer policy). UAV barriers order same-queue writes;
        // the queue fence orders the compute -> graphics handoff. The per-frame
        // GpuScene constants/lights shared read-only across queues are upload-heap
        // (permanently GENERIC_READ) and need no transition.
        //
        // The one operation that DID cause device removal was a spurious wait,
        // not a barrier: the executor made the compute queue wait on the
        // graphics-queue upload fence even though no compute pass consumes
        // uploaded data. That wait is now correctly scoped to the graphics
        // (consumer) queue only. See DX12FrameExecutor::submitAndPresent.
        bool supportsAsyncCompute() const override { return true; }
        void drainForSchedulingTransition() override;

        std::span<const GpuPassSample> gpuPassSamples() const override
        {
            return m_gpuProfiler.lastFrameSamples();
        }
        void setGpuProfilingEnabled(bool enabled) override
        {
            m_gpuProfiler.setEnabled(enabled);
            buildBackendDiagnostics();
        }
        bool gpuProfilingEnabled() const override
        {
            return m_gpuProfiler.enabled();
        }

        HiZDebugImage hiZDebugImage(bool previous, uint32_t mip) override;
        BackendDiagnosticInfo backendDiagnostics() const override;
        RendererPerformanceCounters performanceCounters() const override
        {
            return m_performanceCounters;
        }

        DX12ResourceAllocator& resourceAllocator()
        {
            return m_resourceAllocator;
        }

        const DX12ResourceAllocator& resourceAllocator() const
        {
            return m_resourceAllocator;
        }

    private:
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

        // Clustered-forward derived state. The cluster bounds/light-grid/index/
        // counter buffers are frame-graph owned (materialized by the registry
        // from clustered_forward.h and bound via clusterBufferAddress). There is no
        // duplicate backend copy or hand-rolled state lives here. The GPU-driven
        // cull/indirect-draw buffers live in DX12GpuScene (m_gpuScene).
        struct ClusteredForwardResources
        {
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

        // Constructs the lightweight per-pass context handed to the recorders:
        // command list, plan/node, frame + scene views, and narrow references to
        // the shared subsystems plus the swapchain surface. Backend-private
        // per-technique resources are NOT exposed here. Each wrapper resolves
        // those into a pass-specific *Inputs struct.
        DX12PassContext makePassContext(
            const CompiledGraphPlan& plan,
            const ExecutionNode& node,
            const FrameContext& ctx,
            const SceneRenderView& scene,
            ID3D12GraphicsCommandList4* cmd,
            ID3D12Resource* swapchainImage);

        // Per-payload recorder entry points. dispatchNode std::visits the node's
        // PassPayload variant onto this overload set, so the dispatch is
        // exhaustive by construction: adding a new PassPayload alternative
        // without a matching overload fails to compile, rather than silently
        // no-oping the way the previous get_if chain did. Each overload takes the
        // uniform (plan, node, ctx, scene, cmd, swapchainImage) tuple even when a
        // given pass ignores part of it, so the visit lambda stays a single
        // generic call. The recorders own their own resource lookup, pipeline
        // state, native binding and draw/dispatch; the payload-typed overload is
        // the one place that names each pass, replacing the old node.type switch
        // plus the centralized compute-node get_if cascade.
        void recordPassPayload(
            const GraphicsPassData& payload,
            const CompiledGraphPlan& plan, const ExecutionNode& node,
            const FrameContext& ctx, const SceneRenderView& scene,
            ID3D12GraphicsCommandList4* cmd, ID3D12Resource* swapchainImage);
        void recordPassPayload(
            const ComputePassData& payload,
            const CompiledGraphPlan& plan, const ExecutionNode& node,
            const FrameContext& ctx, const SceneRenderView& scene,
            ID3D12GraphicsCommandList4* cmd, ID3D12Resource* swapchainImage);
        void recordPassPayload(
            const PathTracePassData& payload,
            const CompiledGraphPlan& plan, const ExecutionNode& node,
            const FrameContext& ctx, const SceneRenderView& scene,
            ID3D12GraphicsCommandList4* cmd, ID3D12Resource* swapchainImage);
        void recordPassPayload(
            const TonemapPassData& payload,
            const CompiledGraphPlan& plan, const ExecutionNode& node,
            const FrameContext& ctx, const SceneRenderView& scene,
            ID3D12GraphicsCommandList4* cmd, ID3D12Resource* swapchainImage);
        void recordPassPayload(
            const EnvironmentConvertPassData& payload,
            const CompiledGraphPlan& plan, const ExecutionNode& node,
            const FrameContext& ctx, const SceneRenderView& scene,
            ID3D12GraphicsCommandList4* cmd, ID3D12Resource* swapchainImage);
        void recordPassPayload(
            const TransferPassData& payload,
            const CompiledGraphPlan& plan, const ExecutionNode& node,
            const FrameContext& ctx, const SceneRenderView& scene,
            ID3D12GraphicsCommandList4* cmd, ID3D12Resource* swapchainImage);
        // Declared-but-unused PassPayload alternatives. They keep the visit
        // exhaustive and record nothing (the graph still emits their barriers and
        // queue ordering); collapse them into real recorders when a pass adopts
        // one. GeometryPassData/LightingPassData/ShadowPassData/PostProcessPassData
        // are forward-declared feature payloads; ClearPassData and PresentPassData
        // are handled entirely by graph barriers / the frame executor.
        void recordPassPayload(
            const GeometryPassData&, const CompiledGraphPlan&,
            const ExecutionNode&, const FrameContext&, const SceneRenderView&,
            ID3D12GraphicsCommandList4*, ID3D12Resource*) {}
        void recordPassPayload(
            const LightingPassData&, const CompiledGraphPlan&,
            const ExecutionNode&, const FrameContext&, const SceneRenderView&,
            ID3D12GraphicsCommandList4*, ID3D12Resource*) {}
        void recordPassPayload(
            const ShadowPassData&, const CompiledGraphPlan&,
            const ExecutionNode&, const FrameContext&, const SceneRenderView&,
            ID3D12GraphicsCommandList4*, ID3D12Resource*) {}
        void recordPassPayload(
            const PostProcessPassData&, const CompiledGraphPlan&,
            const ExecutionNode&, const FrameContext&, const SceneRenderView&,
            ID3D12GraphicsCommandList4*, ID3D12Resource*) {}
        void recordPassPayload(
            const ClearPassData&, const CompiledGraphPlan&,
            const ExecutionNode&, const FrameContext&, const SceneRenderView&,
            ID3D12GraphicsCommandList4*, ID3D12Resource*) {}
        void recordPassPayload(
            const PresentPassData&, const CompiledGraphPlan&,
            const ExecutionNode&, const FrameContext&, const SceneRenderView&,
            ID3D12GraphicsCommandList4*, ID3D12Resource*) {}

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
        // Fills the diagnostic feature/limit tables once at init so
        // backendDiagnostics() is a free span read per frame.
        void buildBackendDiagnostics();
        void ensureComputeTestBuffer();
        void ensureClusteredForwardResources();
        bool bindClusteredForwardCompute(
            const DX12ComputePipeline& pipeline,
            const CompiledGraphPlan& plan,
            const FrameContext& ctx,
            const SceneRenderView& scene,
            ID3D12GraphicsCommandList4* cmd);
        // Resolves the graph-registry entry for a GPU-driven buffer by its
        // semantic (current frame slot). Returns nullptr if the graph does not
        // declare it or it is not materialized.
        DX12GraphResourceEntry* gpuDrivenBufferEntry(
            const CompiledGraphPlan& plan,
            GraphResourceSemantic semantic);
        // Uploads this frame's prepared cull inputs (instance bounds, draw
        // inputs) straight into the graph-owned per-frame-slot buffers.
        void uploadGpuDrivenInputs(
            const CompiledGraphPlan& plan,
            const FrameContext& ctx);
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

        D3D12_RESOURCE_STATES usageToState(ResourceUsage usage) const;

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
        GpuOcclusionHistoryState m_gpuOcclusionHistory;
        PathTraceResources m_pathTraceResources;
        EnvironmentResources m_environmentResources;

        const PipelineLibrary* m_pipelineLibrary = nullptr;
        std::unordered_map<AssetHandle, DX12UploadedModel, AssetHandleHash> m_uploadedModels;
        std::unordered_map<uint64_t, UploadedTexture> m_uploadedTextures;
        std::unordered_map<uint64_t, UploadedSampler> m_uploadedSamplers;
        DX12GraphResourceRegistry m_graphResourceRegistry;
        DX12GpuScene m_gpuScene;
        DX12UploadScheduler m_uploadScheduler;
        DX12RetirementQueue m_retirementQueue;

        DX12FrameExecutor m_frameExecutor;
        // Authoritative render-surface generation, bumped on every swapchain
        // (re)creation. The renderer rebuilds the graph whenever this changes.
        uint64_t m_swapchainGeneration = 1;
        uint32_t m_lastFramebufferWidth = 0;
        uint32_t m_lastFramebufferHeight = 0;
        uint32_t m_workerSlots = 1;
        bool m_imguiEnabled = false;
        bool m_imguiFrameActive = false;
        bool m_clusteredForwardHeatmapEnabled = false;
        bool m_hiZDebugViewEnabled = false;
        bool m_hiZDebugPrevious = false;
        bool m_gpuOcclusionEnabled = true;
        GpuCullDebugMode m_gpuCullDebugMode = GpuCullDebugMode::Off;
        GpuCullStats m_gpuCullStats = {};
        GpuCullPerformance m_gpuCullPerformance = {};
        uint64_t m_gpuCullDiagnosticFrames = 0;
        Microsoft::WRL::ComPtr<ID3D12QueryHeap> m_gpuCullTimestampHeap;
        DX12Buffer m_gpuCullTimestampReadback;
        std::vector<double> m_gpuCullCpuRecordMilliseconds;
        std::vector<uint8_t> m_gpuCullTimestampValid;
        uint64_t m_gpuCullTimestampFrequency = 0;
        DX12GpuProfiler m_gpuProfiler;
        std::vector<ID3D12CommandList*> m_frameCommandLists;
        RendererPerformanceCounters m_performanceCounters{};
        // Built once at init; backendDiagnostics() returns spans into these.
        std::string m_diagnosticAdapterName;
        std::vector<BackendFeature> m_diagnosticFeatures;
        std::vector<BackendLimit> m_diagnosticLimits;
        uint32_t m_hiZDebugMip = 0;
        std::string m_imguiIniPath;
	};
}
