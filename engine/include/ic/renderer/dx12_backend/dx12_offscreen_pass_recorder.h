#pragma once

// Recorders for the offscreen compute passes: HDR equirect->cubemap conversion,
// the path tracer, and its tonemap. Each owns its GPU constants build, native
// bindings, dispatch, and the pass-internal UAV barrier. The backend owns all
// lifetime (ensure/destroy/upload of the path-trace and environment resource
// structs), cross-frame accumulation state, and the state-tracked transitions,
// resolving everything the recorder needs into the plain *Inputs structs below.

#include "ic/renderer/dx12_backend/dx12_pass_recorders.h"
#include "ic/renderer/dx12_backend/dx12_pipeline_manager.h"

#include <d3d12.h>
#include <cstdint>

namespace ic
{
    // HDR equirectangular source -> cubemap face conversion.
    struct DX12EnvironmentConvertInputs
    {
        const DX12ComputePipeline* pipeline = nullptr;
        D3D12_GPU_DESCRIPTOR_HANDLE sourceSrv = {};
        D3D12_GPU_DESCRIPTOR_HANDLE cubemapUav = {};
        D3D12_GPU_DESCRIPTOR_HANDLE sampler = {};
        ID3D12Resource* cubemap = nullptr; // pass-internal UAV barrier target
        uint32_t cubemapSize = 0;
    };

    // Records descriptor-heap/root-signature/PSO setup, the 6-face dispatch, and
    // the trailing UAV barrier. Returns false without recording when the pipeline
    // is incomplete (backend logs and leaves the environment unconverted).
    bool recordEnvironmentConvert(
        const DX12PassContext& ctx,
        const DX12EnvironmentConvertInputs& inputs);

    // Path-trace accumulation dispatch. Scalar fields feed the GPU constants the
    // recorder assembles (camera + point lights are read from ctx.scene).
    struct DX12PathTraceInputs
    {
        const DX12ComputePipeline* pipeline = nullptr;
        void* constantsMapped = nullptr;
        D3D12_GPU_VIRTUAL_ADDRESS constantsAddr = 0;
        D3D12_GPU_DESCRIPTOR_HANDLE accumulationUav = {};
        bool sceneSrvsValid = false;
        D3D12_GPU_DESCRIPTOR_HANDLE sceneSrvs = {};
        bool samplerValid = false;
        D3D12_GPU_DESCRIPTOR_HANDLE sampler = {};
        ID3D12Resource* accumulation = nullptr; // pass-internal UAV barrier

        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t frameIndex = 0;
        uint32_t accumulatedSampleCount = 0;
        float exposure = 1.0f;
        bool resetAccumulation = false;
        uint32_t sceneVertexCount = 0;
        uint32_t sceneMaterialCount = 0;
        uint32_t sceneTriangleCount = 0;
        uint32_t sceneBvhNodeCount = 0;
        uint32_t firstEmissiveTriangleIndex = 0;
        bool environmentReady = false;
        float environmentIntensity = 0.0f;
        float environmentExposure = 0.0f;
    };

    // Records the path-trace constants upload, bindings, dispatch, and UAV
    // barrier. Returns false without recording when the pipeline is incomplete.
    bool recordPathTrace(
        const DX12PassContext& ctx,
        const DX12PathTraceInputs& inputs);

    // Tonemap of the accumulation buffer into the LDR tonemap target.
    struct DX12TonemapInputs
    {
        const DX12ComputePipeline* pipeline = nullptr;
        void* constantsMapped = nullptr;
        D3D12_GPU_VIRTUAL_ADDRESS constantsAddr = 0;
        D3D12_GPU_DESCRIPTOR_HANDLE tonemapUav = {};
        D3D12_GPU_DESCRIPTOR_HANDLE accumulationSrv = {};
        ID3D12Resource* tonemap = nullptr; // pass-internal UAV barrier target
        uint32_t width = 0;
        uint32_t height = 0;
        float exposure = 1.0f;
    };

    void recordTonemap(
        const DX12PassContext& ctx,
        const DX12TonemapInputs& inputs);
}
