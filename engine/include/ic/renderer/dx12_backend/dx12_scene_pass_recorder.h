#pragma once

// Recorders for the main render-path graphics/compute passes: the clustered
// light-culling compute binding, the forward/depth scene pass (generic and
// clustered binding layouts), and the skybox. Each owns its graph-resource
// resolution (cluster buffers by semantic, indirect-draw buffers), native
// bindings, and draw/dispatch. The backend owns lifetime (ensure/prepare),
// render-target/viewport setup, attachment clears, and the state-tracked
// transitions, resolving its private per-technique handles into the *Inputs
// structs below.

#include "ic/renderer/dx12_backend/dx12_pass_recorders.h"
#include "ic/renderer/dx12_backend/dx12_pipeline_manager.h"
#include "ic/renderer/dx12_backend/dx12_gpu_scene.h"

#include <d3d12.h>
#include <cstdint>

namespace ic
{
    // Resolves the GPU address of the frame-graph-owned buffer carrying the given
    // cluster semantic (0 if the graph does not declare it). Shared by the
    // clustered compute and graphics recorders.
    D3D12_GPU_VIRTUAL_ADDRESS clusterBufferAddress(
        const DX12PassContext& ctx,
        GraphResourceSemantic semantic);

    // Binds the clustered light-culling compute pass: frame constants plus the
    // frame-graph-owned cluster bounds/grid/indices/counter and visible-lights
    // buffers (root descriptors). The caller has already bound the root
    // signature + PSO and issues the dispatch; this only rebinds the descriptor
    // heaps, root signature, and the cluster root descriptors, matching the
    // original bind order exactly.
    void recordClusteredForwardCompute(
        const DX12PassContext& ctx,
        const DX12ComputePipeline& pipeline,
        D3D12_GPU_VIRTUAL_ADDRESS frameConstantsAddr);

    // Everything the forward/depth scene pass binds and draws. The backend
    // resolves the per-frame-slot GPU-scene handles, the baked-IBL descriptors,
    // and the GPU-driven indirect stream (all backend/registry lifetime) into
    // these plain fields; the recorder owns the heap/root-signature/PSO setup,
    // the clustered-vs-generic binding decision, and the geometry draw loop.
    struct DX12ForwardSceneInputs
    {
        const DX12GraphicsPipeline* pipeline = nullptr;
        bool usesClusterData = false;

        D3D12_GPU_VIRTUAL_ADDRESS frameConstantsAddr = 0;
        D3D12_GPU_DESCRIPTOR_HANDLE objectSrv = {};
        D3D12_GPU_DESCRIPTOR_HANDLE materialSrv = {};

        bool iblBaked = false;
        D3D12_GPU_DESCRIPTOR_HANDLE irradianceSrv = {};
        D3D12_GPU_DESCRIPTOR_HANDLE prefilteredSrv = {};
        D3D12_GPU_DESCRIPTOR_HANDLE brdfLutSrv = {};
        D3D12_GPU_DESCRIPTOR_HANDLE diffuseGiSrv = {};
        D3D12_GPU_DESCRIPTOR_HANDLE environmentSampler = {};

        bool useGpuDriven = false;
        DX12IndirectDrawStream indirectStream = {};
        std::span<const DX12GpuScene::DrawItem> draws = {};
        std::span<const DX12GpuScene::GeometryBin> geometryBins = {};
        DX12ResolveNativeModelFn resolveNativeModel = {};
    };

    void recordForwardScene(
        const DX12PassContext& ctx,
        const DX12ForwardSceneInputs& inputs);

    // Skybox fullscreen-triangle draw. The backend resolves the per-frame skybox
    // constants buffer and cubemap/sampler descriptors; the recorder assembles
    // the constants from ctx.scene + surface size, binds, and draws.
    struct DX12SkyboxInputs
    {
        const DX12GraphicsPipeline* pipeline = nullptr;
        void* constantsMapped = nullptr;
        D3D12_GPU_VIRTUAL_ADDRESS constantsAddr = 0;
        D3D12_GPU_DESCRIPTOR_HANDLE cubemapSrv = {};
        D3D12_GPU_DESCRIPTOR_HANDLE sampler = {};
    };

    void recordSkybox(
        const DX12PassContext& ctx,
        const DX12SkyboxInputs& inputs);
}
