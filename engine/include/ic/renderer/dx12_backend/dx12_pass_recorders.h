#pragma once

#include "ic/renderer/dx12_backend/dx12_descriptor_system.h"
#include "ic/renderer/dx12_backend/dx12_graph_resource_registry.h"
#include "ic/renderer/dx12_backend/dx12_gpu_scene.h"
#include "ic/renderer/dx12_backend/dx12_pipeline_manager.h"
#include "ic/renderer/frame_graph/compiled_graph_plan.h"

#include <d3d12.h>
#include <cstdint>
#include <functional>
#include <span>

namespace ic
{
    struct FrameContext;
    struct SceneRenderView;
    class DX12PipelineManager;
    class DX12GpuScene;

    // Lightweight, API-specific pass context handed to the DX12 pass recorders.
    // It carries the native command list, the executing node/plan, per-frame
    // state, and narrow references to the SHARED subsystems a recorder reads to
    // resolve graph resources and bind pipelines/descriptors. It deliberately
    // does NOT own the device, swapchain lifecycle, queue submission, or any
    // backend-private per-technique resource struct (those the backend resolves
    // into a pass-specific *Inputs struct before invoking the recorder, and it
    // performs any "ensure"/prepare and state-tracked barriers itself). Passed by
    // const&; no virtual dispatch on the recording hot path.
    struct DX12PassContext
    {
        ID3D12GraphicsCommandList4* cmd = nullptr;
        const CompiledGraphPlan* plan = nullptr;
        const ExecutionNode* node = nullptr;
        const FrameContext* frame = nullptr;
        const SceneRenderView* scene = nullptr;

        DX12GraphResourceRegistry* resources = nullptr;
        DX12DescriptorSystem* descriptors = nullptr;
        DX12PipelineManager* pipelines = nullptr;
        DX12GpuScene* gpuScene = nullptr;

        // Imported swapchain backbuffer for this frame plus its dimensions, so a
        // recorder can resolve the imported color target and set viewport/scissor
        // without reaching into the swapchain object. Fallback RTV/DSV are the
        // backend's own targets, used only when the graph declares no attachment.
        ID3D12Resource* swapchainImage = nullptr;
        uint32_t surfaceWidth = 0;
        uint32_t surfaceHeight = 0;
        D3D12_CPU_DESCRIPTOR_HANDLE fallbackRtv = {};
        D3D12_CPU_DESCRIPTOR_HANDLE fallbackDsv = {};
    };

    // Native, already-resolved source/destination for a transfer-copy pass. The
    // backend resolves both (including imported swapchain and backend-private
    // path-trace targets) and applies any state-tracked transition first; the
    // recorder owns only the type/description validation and the copy command.
    struct DX12TransferCopy
    {
        ID3D12Resource* source = nullptr;
        ID3D12Resource* destination = nullptr;
        GraphResourceType type = GraphResourceType::Texture;
        const char* passName = "";
    };

    void recordTransferCopy(
        ID3D12GraphicsCommandList4* cmd,
        const DX12TransferCopy& copy);

    // Diagnostic storage-buffer compute pass. The backend has already bound the
    // root signature + PSO and ensured/transitioned the buffer; the recorder
    // binds the UAV, dispatches, and issues the trailing UAV barrier.
    struct DX12ComputeStorageBufferTest
    {
        D3D12_GPU_VIRTUAL_ADDRESS bufferAddr = 0;
        ID3D12Resource* bufferResource = nullptr;
        uint32_t groupCountX = 1;
        uint32_t groupCountY = 1;
        uint32_t groupCountZ = 1;
    };

    void recordComputeStorageBufferTest(
        const DX12PassContext& ctx,
        const DX12ComputeStorageBufferTest& test);

    // Native GPU addresses of the GPU-driven cull buffers. Inputs and outputs
    // alike are graph-registry-owned; the backend resolves each by semantic and
    // fills this. State/queue ordering is owned by the frame graph's barriers
    // (buffers rely on D3D12 implicit promotion/decay), so the recorder no
    // longer transitions anything. It only binds addresses.
    struct DX12CullBuffers
    {
        D3D12_GPU_VIRTUAL_ADDRESS visibleInstancesAddr = 0;
        D3D12_GPU_VIRTUAL_ADDRESS visibleInstanceCountAddr = 0;
        D3D12_GPU_VIRTUAL_ADDRESS indirectArgumentsAddr = 0;
        D3D12_GPU_VIRTUAL_ADDRESS binCountsAddr = 0;

        // Cull compute inputs (frame constants + per-instance bounds + draw
        // inputs). frameConstants is the backend per-frame CB; instanceBounds
        // and drawInputs are graph-owned per-frame-slot upload buffers.
        D3D12_GPU_VIRTUAL_ADDRESS frameConstantsAddr = 0;
        D3D12_GPU_VIRTUAL_ADDRESS instanceBoundsAddr = 0;
        D3D12_GPU_VIRTUAL_ADDRESS drawInputsAddr = 0;
        D3D12_GPU_DESCRIPTOR_HANDLE previousHiZSrv = {};
        D3D12_GPU_VIRTUAL_ADDRESS cullClassificationAddr = 0;
        D3D12_GPU_VIRTUAL_ADDRESS cullStatsAddr = 0;
    };

    // Native depth + Hi-Z inputs for the Hi-Z pyramid recorder.
    struct DX12HiZInputs
    {
        GraphResourceId hiZId = InvalidGraphResourceId;
        GraphResourceId sceneDepthId = InvalidGraphResourceId;
        D3D12_GPU_VIRTUAL_ADDRESS frameConstantsAddr = 0;
        // Set to hiZId once the Hi-Z resource is validated (drives the debug
        // overlay), preserving the original ordering even on later bail-outs.
        GraphResourceId* hiZDebugResourceOut = nullptr;
    };

    // Records the Hi-Z depth-pyramid mip chain. Returns true when it recorded
    // (so the backend can mark the Hi-Z debug resource). Samples mip 0 from the
    // sceneDepth graph resource's own SRV. The frame graph already transitions
    // it to SampledTexture before this node runs, via the normal barrier path
    // (this pass declares .read(sceneDepth, ResourceUsage::SampledTexture)).
    bool recordHiZPyramid(
        const DX12PassContext& ctx,
        const DX12ComputePipeline& pipeline,
        const DX12HiZInputs& inputs);

    // Records the GPU frustum-cull + command-build dispatch bindings (the
    // dispatch itself is issued by the caller). Binds addresses only; the frame
    // graph owns the buffers' state and cross-pass/queue ordering.
    void recordGpuFrustumCull(
        const DX12PassContext& ctx,
        const DX12CullBuffers& buffers);

    struct DX12OcclusionValidationInputs
    {
        D3D12_GPU_VIRTUAL_ADDRESS frameConstantsAddr = 0;
        D3D12_GPU_VIRTUAL_ADDRESS instanceBoundsAddr = 0;
        D3D12_GPU_VIRTUAL_ADDRESS cullClassificationAddr = 0;
        D3D12_GPU_VIRTUAL_ADDRESS cullStatsAddr = 0;
        D3D12_GPU_DESCRIPTOR_HANDLE currentHiZSrv = {};
    };

    void recordGpuOcclusionValidation(
        const DX12PassContext& ctx,
        const DX12OcclusionValidationInputs& inputs);

    // Native indirect-draw stream consumed by GPU-driven ExecuteIndirect.
    struct DX12IndirectDrawStream
    {
        ID3D12CommandSignature* commandSignature = nullptr;
        ID3D12Resource* indirectArguments = nullptr;
        ID3D12Resource* binCounts = nullptr;
        uint32_t commandStride = 0;
    };

    // Resolves a scene draw's AssetHandle to its uploaded native model
    // (vertex/index buffers). Draws/bins are grouped contiguously by model
    // (PreparedGpuScene sorts by model), so this is called once per bin
    // transition, not once per draw.
    using DX12ResolveNativeModelFn =
        std::function<DX12UploadedModel*(AssetHandle)>;

    // Records the per-draw vertex/index-buffer binding + root-constant +
    // DrawIndexedInstanced submission loop for a prepared scene draw list, or
    // the ExecuteIndirect-per-bin path when useGpuDriven is set. This is the
    // draw-submission body shared by the depth prepass and the forward scene
    // pass alike. Both pipelines route scene geometry through this same
    // recorder, differing only in bound root signature/pixel output, which
    // the caller sets up beforehand. Root parameter 3 is the fallback/per-draw
    // root-constant slot, matching the bound pipeline's layout.
    void recordSceneGeometryDraws(
        ID3D12GraphicsCommandList4* cmd,
        std::span<const DX12GpuScene::DrawItem> draws,
        std::span<const DX12GpuScene::GeometryBin> geometryBins,
        bool useGpuDriven,
        const DX12IndirectDrawStream& indirectStream,
        const DX12ResolveNativeModelFn& resolveNativeModel);
}
