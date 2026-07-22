#include "ic/renderer/dx12_backend/dx12_offscreen_pass_recorder.h"
#include "ic/renderer/dx12_backend/dx12_descriptor_system.h"

#include "ic/core/frame_context.h"
#include "ic/scene/scene_render_view.h"
#include "ic/scene/scene_components.h"
#include "ic/renderer/renderer_common/renderer_util.h"

#include <spdlog/spdlog.h>

#include <cstring>

namespace ic
{
    bool recordEnvironmentConvert(
        const DX12PassContext& ctx,
        const DX12EnvironmentConvertInputs& inputs)
    {
        if (!inputs.pipeline ||
            !inputs.pipeline->rootSignature ||
            !inputs.pipeline->pipelineState)
        {
            spdlog::error(
                "[DX12] Environment conversion pipeline is incomplete; skipping HDR conversion");
            return false;
        }

        ID3D12GraphicsCommandList4* cmd = ctx.cmd;
        ID3D12DescriptorHeap* heaps[] =
        {
            ctx.descriptors->shaderResourceHeap(),
            ctx.descriptors->samplerHeap()
        };
        cmd->SetDescriptorHeaps(
            static_cast<UINT>(std::size(heaps)),
            heaps);
        cmd->SetComputeRootSignature(inputs.pipeline->rootSignature.Get());
        cmd->SetPipelineState(inputs.pipeline->pipelineState.Get());
        cmd->SetComputeRootDescriptorTable(0, inputs.sourceSrv);
        cmd->SetComputeRootDescriptorTable(1, inputs.cubemapUav);
        cmd->SetComputeRootDescriptorTable(2, inputs.sampler);
        cmd->Dispatch(
            (inputs.cubemapSize + 7u) / 8u,
            (inputs.cubemapSize + 7u) / 8u,
            6u);

        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = inputs.cubemap;
        cmd->ResourceBarrier(1, &barrier);
        return true;
    }

    bool recordPathTrace(
        const DX12PassContext& ctx,
        const DX12PathTraceInputs& inputs)
    {
        if (!inputs.pipeline ||
            !inputs.pipeline->rootSignature ||
            !inputs.pipeline->pipelineState)
        {
            spdlog::error(
                "[DX12] Path trace pipeline is incomplete; skipping dispatch");
            return false;
        }

        const SceneRenderView& scene = *ctx.scene;

        PathTraceConstants constants{};
        constants.renderWidth = inputs.width;
        constants.renderHeight = inputs.height;
        constants.frameIndex = inputs.frameIndex;
        constants.accumulatedSampleCount = inputs.accumulatedSampleCount;
        constants.exposure = inputs.exposure;
        constants.resetAccumulation = inputs.resetAccumulation ? 1u : 0u;
        constants.maxBounces = configuredPathTraceMaxBounces();
        constants.samplesPerPixel = DefaultPathTraceSamplesPerPixel;
        constants.sceneVertexCount = inputs.sceneVertexCount;
        constants.sceneMaterialCount = inputs.sceneMaterialCount;
        constants.sceneTriangleCount = inputs.sceneTriangleCount;
        constants.sceneBvhNodeCount = inputs.sceneBvhNodeCount;
        constants.sceneEmissiveTriangleIndex = inputs.firstEmissiveTriangleIndex;
        constants.sceneEmissiveTriangleCount = inputs.emissiveTriangleCount;
        constants.referenceMode = configuredPathTraceReferenceMode();
        constants.useSceneGeometry =
            inputs.sceneTriangleCount != 0u &&
            inputs.sceneBvhNodeCount != 0u
                ? 1u
                : 0u;
        constants.environmentEnabled = inputs.environmentReady ? 1u : 0u;
        constants.environmentIntensity = inputs.environmentIntensity;
        constants.environmentExposure = inputs.environmentExposure;
        fillPathTraceCameraConstants(
            scene.camera,
            inputs.width,
            inputs.height,
            constants);
        for (const SceneLightRenderItem& light : scene.lights)
        {
            if (light.type != LightType::Point ||
                constants.pointLightCount >= MaxPathTracePointLights)
            {
                continue;
            }

            const uint32_t lightIndex = constants.pointLightCount++;
            constants.pointLightPositionRange[lightIndex] =
                glm::vec4(light.position, light.range);
            constants.pointLightColorIntensity[lightIndex] =
                glm::vec4(light.color, light.intensity);
        }

        std::memcpy(
            inputs.constantsMapped,
            &constants,
            sizeof(constants));

        ID3D12GraphicsCommandList4* cmd = ctx.cmd;
        ID3D12DescriptorHeap* heaps[] =
        {
            ctx.descriptors->shaderResourceHeap(),
            ctx.descriptors->samplerHeap()
        };
        cmd->SetDescriptorHeaps(
            static_cast<UINT>(std::size(heaps)),
            heaps);

        cmd->SetComputeRootSignature(inputs.pipeline->rootSignature.Get());
        cmd->SetPipelineState(inputs.pipeline->pipelineState.Get());
        cmd->SetComputeRootConstantBufferView(0, inputs.constantsAddr);
        cmd->SetComputeRootDescriptorTable(1, inputs.accumulationUav);
        if (inputs.sceneSrvsValid)
        {
            cmd->SetComputeRootDescriptorTable(2, inputs.sceneSrvs);
        }
        if (inputs.samplerValid)
        {
            cmd->SetComputeRootDescriptorTable(3, inputs.sampler);
        }
        cmd->SetComputeRootDescriptorTable(
            4,
            ctx.descriptors->shaderResourceGpuStart());
        cmd->SetComputeRootDescriptorTable(
            5,
            ctx.descriptors->samplerGpuStart());

        const uint32_t groupCountX = (inputs.width + 7u) / 8u;
        const uint32_t groupCountY = (inputs.height + 7u) / 8u;
        cmd->Dispatch(groupCountX, groupCountY, 1);

        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = inputs.accumulation;
        cmd->ResourceBarrier(1, &barrier);
        return true;
    }

    void recordTonemap(
        const DX12PassContext& ctx,
        const DX12TonemapInputs& inputs)
    {
        TonemapConstants constants{};
        constants.renderWidth = inputs.width;
        constants.renderHeight = inputs.height;
        constants.exposure = inputs.exposure;

        std::memcpy(
            inputs.constantsMapped,
            &constants,
            sizeof(constants));

        ID3D12GraphicsCommandList4* cmd = ctx.cmd;
        ID3D12DescriptorHeap* heaps[] =
        {
            ctx.descriptors->shaderResourceHeap()
        };
        cmd->SetDescriptorHeaps(1, heaps);
        cmd->SetComputeRootSignature(inputs.pipeline->rootSignature.Get());
        cmd->SetPipelineState(inputs.pipeline->pipelineState.Get());
        cmd->SetComputeRootConstantBufferView(0, inputs.constantsAddr);
        cmd->SetComputeRootDescriptorTable(1, inputs.tonemapUav);
        cmd->SetComputeRootDescriptorTable(2, inputs.accumulationSrv);

        const uint32_t groupCountX = (inputs.width + 7u) / 8u;
        const uint32_t groupCountY = (inputs.height + 7u) / 8u;
        cmd->Dispatch(groupCountX, groupCountY, 1);

        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = inputs.tonemap;
        cmd->ResourceBarrier(1, &barrier);
    }
}
