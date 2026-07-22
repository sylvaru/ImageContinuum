#include "ic/renderer/dx12_backend/dx12_scene_pass_recorder.h"
#include "ic/renderer/dx12_backend/dx12_descriptor_system.h"

#include "ic/core/frame_context.h"
#include "ic/scene/scene_render_view.h"
#include "ic/renderer/renderer_gpu_assets.h"
#include "ic/renderer/renderer_common/renderer_util.h"

#include <cstring>
#include <iterator>

namespace ic
{
    namespace
    {
        // Clustered-forward graphics binding: reads the frame-graph-owned cluster
        // buffers through root SRV descriptors. State is owned by the frame graph
        // and the compute->graphics handoff by the queue fence; no transition is
        // recorded here. Distinct root layout from the generic path (no sampler
        // table 5; extra cluster SRVs 13/15/16/17), preserved verbatim.
        void recordClusteredForwardGraphics(
            const DX12PassContext& ctx,
            const DX12ForwardSceneInputs& inputs)
        {
            ID3D12GraphicsCommandList4* cmd = ctx.cmd;
            cmd->SetGraphicsRootSignature(
                inputs.pipeline->rootSignature.Get());
            cmd->SetGraphicsRootConstantBufferView(0, inputs.frameConstantsAddr);
            cmd->SetGraphicsRootDescriptorTable(1, inputs.objectSrv);
            cmd->SetGraphicsRootDescriptorTable(2, inputs.materialSrv);
            cmd->SetGraphicsRootDescriptorTable(
                4,
                ctx.descriptors->shaderResourceGpuStart());
            cmd->SetGraphicsRootDescriptorTable(
                5,
                ctx.descriptors->samplerGpuStart());

            if (inputs.iblBaked)
            {
                cmd->SetGraphicsRootDescriptorTable(6, inputs.irradianceSrv);
                cmd->SetGraphicsRootDescriptorTable(7, inputs.prefilteredSrv);
                cmd->SetGraphicsRootDescriptorTable(8, inputs.brdfLutSrv);
                cmd->SetGraphicsRootDescriptorTable(9, inputs.environmentSampler);
            }
            if (inputs.diffuseGiSrv.ptr != 0)
            {
                cmd->SetGraphicsRootDescriptorTable(18, inputs.diffuseGiSrv);
            }
            cmd->SetGraphicsRootShaderResourceView(
                13,
                clusterBufferAddress(ctx, GraphResourceSemantic::VisibleLights));
            cmd->SetGraphicsRootShaderResourceView(
                15,
                clusterBufferAddress(ctx, GraphResourceSemantic::ClusterBounds));
            cmd->SetGraphicsRootShaderResourceView(
                16,
                clusterBufferAddress(
                    ctx, GraphResourceSemantic::ClusterLightGrid));
            cmd->SetGraphicsRootShaderResourceView(
                17,
                clusterBufferAddress(
                    ctx, GraphResourceSemantic::ClusterLightIndices));
        }
    }
    D3D12_GPU_VIRTUAL_ADDRESS clusterBufferAddress(
        const DX12PassContext& ctx,
        GraphResourceSemantic semantic)
    {
        const GraphResourceId id =
            findResourceBySemantic(*ctx.plan, semantic);
        if (id == InvalidGraphResourceId)
        {
            return 0;
        }
        const DX12GraphResourceEntry* entry = ctx.resources->entry(id);
        return entry ? entry->buffer.gpuAddress : 0;
    }

    void recordClusteredForwardCompute(
        const DX12PassContext& ctx,
        const DX12ComputePipeline& pipeline,
        D3D12_GPU_VIRTUAL_ADDRESS frameConstantsAddr)
    {
        ID3D12GraphicsCommandList4* cmd = ctx.cmd;

        // The cluster buffers are the frame-graph-owned (registry) resources,
        // bound through root UAV descriptors. Their state, transitions, cross-
        // queue handoff and lifetime are owned entirely by the frame graph /
        // executor (see recordBarrier's buffer policy). The backend keeps no
        // duplicate copy and issues no manual transitions.
        ID3D12DescriptorHeap* heaps[] =
        {
            ctx.descriptors->shaderResourceHeap(),
            ctx.descriptors->samplerHeap()
        };
        cmd->SetDescriptorHeaps(
            static_cast<UINT>(std::size(heaps)),
            heaps);
        cmd->SetComputeRootSignature(pipeline.rootSignature.Get());
        cmd->SetComputeRootConstantBufferView(0, frameConstantsAddr);

        cmd->SetComputeRootUnorderedAccessView(
            10,
            clusterBufferAddress(ctx, GraphResourceSemantic::ClusterBounds));

        cmd->SetComputeRootUnorderedAccessView(
            11,
            clusterBufferAddress(ctx, GraphResourceSemantic::ClusterLightGrid));

        cmd->SetComputeRootUnorderedAccessView(
            12,
            clusterBufferAddress(
                ctx, GraphResourceSemantic::ClusterLightIndices));

        cmd->SetComputeRootShaderResourceView(
            13,
            clusterBufferAddress(ctx, GraphResourceSemantic::VisibleLights));

        cmd->SetComputeRootUnorderedAccessView(
            14,
            clusterBufferAddress(
                ctx, GraphResourceSemantic::ClusterLightCounter));
    }

    void recordForwardScene(
        const DX12PassContext& ctx,
        const DX12ForwardSceneInputs& inputs)
    {
        ID3D12GraphicsCommandList4* cmd = ctx.cmd;

        ID3D12DescriptorHeap* heaps[] =
        {
            ctx.descriptors->shaderResourceHeap(),
            ctx.descriptors->samplerHeap()
        };
        cmd->SetDescriptorHeaps(
            static_cast<UINT>(std::size(heaps)),
            heaps);

        cmd->SetGraphicsRootSignature(inputs.pipeline->rootSignature.Get());
        cmd->SetPipelineState(inputs.pipeline->pipelineState.Get());
        if (inputs.pipeline->desc.bindingLayout ==
                PipelineBindingLayoutKind::ClusteredForward &&
            inputs.usesClusterData)
        {
            recordClusteredForwardGraphics(ctx, inputs);
        }
        else
        {
            cmd->SetGraphicsRootConstantBufferView(0, inputs.frameConstantsAddr);
            cmd->SetGraphicsRootDescriptorTable(1, inputs.objectSrv);
            cmd->SetGraphicsRootDescriptorTable(2, inputs.materialSrv);
            cmd->SetGraphicsRootDescriptorTable(
                4,
                ctx.descriptors->shaderResourceGpuStart());
            cmd->SetGraphicsRootDescriptorTable(
                5,
                ctx.descriptors->samplerGpuStart());
            if (inputs.iblBaked)
            {
                cmd->SetGraphicsRootDescriptorTable(6, inputs.irradianceSrv);
                cmd->SetGraphicsRootDescriptorTable(7, inputs.prefilteredSrv);
                cmd->SetGraphicsRootDescriptorTable(8, inputs.brdfLutSrv);
                cmd->SetGraphicsRootDescriptorTable(9, inputs.environmentSampler);
            }
        }
        cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

        const DrawConstants unusedFallbackConstants{};
        cmd->SetGraphicsRoot32BitConstants(
            3,
            4,
            &unusedFallbackConstants,
            0);

        // Shared by the depth prepass and the forward pass: both pipelines route
        // scene geometry through this same recorder (see dx12_pass_recorders.h
        // for why there is no separate depth-only path).
        recordSceneGeometryDraws(
            cmd,
            inputs.draws,
            inputs.geometryBins,
            inputs.useGpuDriven,
            inputs.indirectStream,
            inputs.resolveNativeModel);
    }

    void recordSkybox(
        const DX12PassContext& ctx,
        const DX12SkyboxInputs& inputs)
    {
        const SceneRenderView& scene = *ctx.scene;

        SkyboxConstants constants{};
        fillSkyboxConstants(
            scene.camera,
            ctx.surfaceWidth,
            ctx.surfaceHeight,
            scene.environment.settings,
            constants);
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
        cmd->SetGraphicsRootSignature(inputs.pipeline->rootSignature.Get());
        cmd->SetPipelineState(inputs.pipeline->pipelineState.Get());
        cmd->SetGraphicsRootConstantBufferView(0, inputs.constantsAddr);
        cmd->SetGraphicsRootDescriptorTable(1, inputs.cubemapSrv);
        cmd->SetGraphicsRootDescriptorTable(2, inputs.sampler);
        cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        cmd->DrawInstanced(3, 1, 0, 0);
    }
}
