#pragma once

#include "ic/renderer/dx12_backend/dx12_descriptor_system.h"
#include "ic/renderer/dx12_backend/dx12_graph_resource_registry.h"
#include "ic/renderer/dx12_backend/dx12_gpu_scene.h"
#include "ic/renderer/dx12_backend/dx12_pipeline_manager.h"
#include "ic/renderer/frame_graph/compiled_graph_plan.h"

#include <d3d12.h>
#include <functional>
#include <span>

namespace ic
{
    // Lightweight, API-specific pass context handed to the DX12 pass recorders.
    // It carries the native command list, the executing node, and the shared
    // subsystems the recorders read. It deliberately does NOT own the device,
    // swapchain, frame lifecycle or queue submission — the backend performs any
    // resource "ensure"/prepare before invoking a recorder. Passed by const&;
    // no virtual dispatch on the recording hot path.
    struct DX12PassContext
    {
        ID3D12GraphicsCommandList4* cmd = nullptr;
        const CompiledGraphPlan* plan = nullptr;
        const ExecutionNode* node = nullptr;
        DX12GraphResourceRegistry* resources = nullptr;
        DX12DescriptorSystem* descriptors = nullptr;
    };

    // Native view of the GPU-driven draw-stream buffers consumed by the cull
    // recorder. The backend fills this from DX12GpuScene, which owns the
    // buffers.
    struct DX12CullBuffers
    {
        ID3D12Resource* visibleInstances = nullptr;
        D3D12_RESOURCE_STATES* visibleInstancesState = nullptr;
        D3D12_GPU_VIRTUAL_ADDRESS visibleInstancesAddr = 0;

        ID3D12Resource* visibleInstanceCount = nullptr;
        D3D12_RESOURCE_STATES* visibleInstanceCountState = nullptr;
        D3D12_GPU_VIRTUAL_ADDRESS visibleInstanceCountAddr = 0;

        ID3D12Resource* indirectArguments = nullptr;
        D3D12_RESOURCE_STATES* indirectArgumentsState = nullptr;
        D3D12_GPU_VIRTUAL_ADDRESS indirectArgumentsAddr = 0;

        ID3D12Resource* binCounts = nullptr;
        D3D12_RESOURCE_STATES* binCountsState = nullptr;
        D3D12_GPU_VIRTUAL_ADDRESS binCountsAddr = 0;

        // Cull compute inputs (frame constants + per-instance bounds + draw
        // inputs), supplied by the current frame's scene resources.
        D3D12_GPU_VIRTUAL_ADDRESS frameConstantsAddr = 0;
        D3D12_GPU_VIRTUAL_ADDRESS instanceBoundsAddr = 0;
        D3D12_GPU_VIRTUAL_ADDRESS drawInputsAddr = 0;
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
    // sceneDepth graph resource's own SRV — the frame graph already transitions
    // it to SampledTexture before this node runs, via the normal barrier path
    // (this pass declares .read(sceneDepth, ResourceUsage::SampledTexture)).
    bool recordHiZPyramid(
        const DX12PassContext& ctx,
        const DX12ComputePipeline& pipeline,
        const DX12HiZInputs& inputs);

    // Records the GPU frustum-cull + command-build dispatch bindings (the
    // dispatch itself is issued by the caller). Transitions the draw-stream
    // buffers to UAV via their mutable states.
    void recordGpuFrustumCull(
        const DX12PassContext& ctx,
        const DX12CullBuffers& buffers);

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
    // pass alike — both pipelines route scene geometry through this same
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
