#include "ic/renderer/dx12_backend/dx12_pass_recorders.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <iterator>

namespace ic
{
    void recordTransferCopy(
        ID3D12GraphicsCommandList4* cmd,
        const DX12TransferCopy& copy)
    {
        if (!copy.source || !copy.destination)
        {
            spdlog::error(
                "Transfer pass '{}' could not resolve its resources",
                copy.passName);
            return;
        }

        const D3D12_RESOURCE_DESC sourceNativeDesc = copy.source->GetDesc();
        const D3D12_RESOURCE_DESC destinationNativeDesc =
            copy.destination->GetDesc();
        if (copy.type == GraphResourceType::Buffer)
        {
            cmd->CopyBufferRegion(
                copy.destination,
                0,
                copy.source,
                0,
                std::min(sourceNativeDesc.Width, destinationNativeDesc.Width));
            return;
        }

        if (sourceNativeDesc.Dimension != destinationNativeDesc.Dimension ||
            sourceNativeDesc.Width != destinationNativeDesc.Width ||
            sourceNativeDesc.Height != destinationNativeDesc.Height ||
            sourceNativeDesc.DepthOrArraySize !=
                destinationNativeDesc.DepthOrArraySize ||
            sourceNativeDesc.MipLevels != destinationNativeDesc.MipLevels ||
            sourceNativeDesc.Format != destinationNativeDesc.Format ||
            sourceNativeDesc.SampleDesc.Count !=
                destinationNativeDesc.SampleDesc.Count)
        {
            spdlog::error(
                "Transfer pass '{}' requires matching texture descriptions",
                copy.passName);
            return;
        }
        cmd->CopyResource(copy.destination, copy.source);
    }

    void recordComputeStorageBufferTest(
        const DX12PassContext& ctx,
        const DX12ComputeStorageBufferTest& test)
    {
        ID3D12GraphicsCommandList4* cmd = ctx.cmd;
        cmd->SetComputeRootUnorderedAccessView(0, test.bufferAddr);
        cmd->Dispatch(test.groupCountX, test.groupCountY, test.groupCountZ);

        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = test.bufferResource;
        cmd->ResourceBarrier(1, &barrier);
    }

    bool recordHiZPyramid(
        const DX12PassContext& ctx,
        const DX12ComputePipeline& pipeline,
        const DX12HiZInputs& inputs)
    {
        DX12GraphResourceEntry* sceneDepth =
            ctx.resources->entry(inputs.sceneDepthId);
        DX12GraphResourceEntry* hiZ = ctx.resources->entry(inputs.hiZId);
        if (!sceneDepth || !hiZ ||
            sceneDepth->mipSrvs.empty() ||
            hiZ->mipSrvs.empty() ||
            hiZ->mipUavs.empty())
        {
            return false;
        }

        if (inputs.hiZDebugResourceOut)
        {
            *inputs.hiZDebugResourceOut = inputs.hiZId;
        }

        if (inputs.frameConstantsAddr == 0)
        {
            return false;
        }

        ID3D12GraphicsCommandList4* cmd = ctx.cmd;

        ID3D12DescriptorHeap* heaps[] =
        {
            ctx.descriptors->shaderResourceHeap(),
            ctx.descriptors->samplerHeap()
        };
        cmd->SetDescriptorHeaps(static_cast<UINT>(std::size(heaps)), heaps);
        cmd->SetComputeRootSignature(pipeline.rootSignature.Get());
        cmd->SetPipelineState(pipeline.pipelineState.Get());
        cmd->SetComputeRootConstantBufferView(0, inputs.frameConstantsAddr);

        ID3D12Resource* const hiZResource = hiZ->texture.resource.Get();
        const uint32_t mipCount = hiZ->mipLevels();

        auto transitionMip =
            [&](uint32_t mip,
                D3D12_RESOURCE_STATES before,
                D3D12_RESOURCE_STATES after)
            {
                D3D12_RESOURCE_BARRIER barrier{};
                barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barrier.Transition.pResource = hiZResource;
                barrier.Transition.Subresource = mip;
                barrier.Transition.StateBefore = before;
                barrier.Transition.StateAfter = after;
                cmd->ResourceBarrier(1, &barrier);
            };

        uint32_t mipWidth = hiZ->width();
        uint32_t mipHeight = hiZ->height();
        for (uint32_t mip = 0; mip < mipCount; ++mip)
        {
            // The downsample reads the previous mip as an SRV. The frame graph
            // brings the whole resource in as UNORDERED_ACCESS, so before the
            // dispatch each source mip must be transitioned to a shader-read
            // state; otherwise the GPU reads a subresource still in UAV state,
            // which is undefined and hangs the device in Release builds.
            cmd->SetComputeRootDescriptorTable(
                1,
                mip == 0
                    ? sceneDepth->mipSrvs[0].gpuStart
                    : hiZ->mipSrvs[mip - 1u].gpuStart);
            cmd->SetComputeRootDescriptorTable(
                2,
                hiZ->mipUavs[mip].gpuStart);
            cmd->Dispatch(
                (mipWidth + 7u) / 8u,
                (mipHeight + 7u) / 8u,
                1);

            // Make this freshly written mip readable as an SRV by the next
            // iteration. The final mip has no consumer inside this loop, so it
            // stays in UNORDERED_ACCESS.
            if (mip + 1u < mipCount)
            {
                transitionMip(
                    mip,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }

            mipWidth = std::max(1u, mipWidth >> 1u);
            mipHeight = std::max(1u, mipHeight >> 1u);
        }

        // Restore every mip that was flipped to a read state back to
        // UNORDERED_ACCESS so the whole resource is uniformly in the state the
        // frame graph tracks it in (StorageTexture / UAV). This keeps the
        // executor's end-of-life transient transition (all-subresources
        // UAV -> COMMON) valid.
        for (uint32_t mip = 0; mip + 1u < mipCount; ++mip)
        {
            transitionMip(
                mip,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }

        return true;
    }

    void recordSceneGeometryDraws(
        ID3D12GraphicsCommandList4* cmd,
        std::span<const DX12GpuScene::DrawItem> draws,
        std::span<const DX12GpuScene::GeometryBin> geometryBins,
        bool useGpuDriven,
        const DX12IndirectDrawStream& indirectStream,
        const DX12ResolveNativeModelFn& resolveNativeModel)
    {
        if (useGpuDriven)
        {
            for (uint32_t binIndex = 0;
                 binIndex < geometryBins.size();
                 ++binIndex)
            {
                const DX12GpuScene::GeometryBin& bin = geometryBins[binIndex];
                if (!bin.model || bin.maxDrawCount == 0u)
                {
                    continue;
                }
                DX12UploadedModel* model = resolveNativeModel(bin.model);
                if (!model)
                {
                    continue;
                }
                D3D12_VERTEX_BUFFER_VIEW vbv{};
                vbv.BufferLocation = model->vertexBuffer.gpuAddress;
                vbv.SizeInBytes = static_cast<UINT>(model->vertexBuffer.size);
                vbv.StrideInBytes = sizeof(AssetVertex);
                D3D12_INDEX_BUFFER_VIEW ibv{};
                ibv.BufferLocation = model->indexBuffer.gpuAddress;
                ibv.SizeInBytes = static_cast<UINT>(model->indexBuffer.size);
                ibv.Format = DXGI_FORMAT_R32_UINT;
                cmd->IASetVertexBuffers(0, 1, &vbv);
                cmd->IASetIndexBuffer(&ibv);
                cmd->ExecuteIndirect(
                    indirectStream.commandSignature,
                    bin.maxDrawCount,
                    indirectStream.indirectArguments,
                    static_cast<UINT64>(bin.commandOffset) *
                        indirectStream.commandStride,
                    indirectStream.binCounts,
                    static_cast<UINT64>(binIndex) * sizeof(uint32_t));
            }
            return;
        }

        bool haveBoundModel = false;
        AssetHandle boundModelHandle{};
        DX12UploadedModel* boundModel = nullptr;
        for (const DX12GpuScene::DrawItem& draw : draws)
        {
            if (!haveBoundModel || draw.model != boundModelHandle)
            {
                boundModel = resolveNativeModel(draw.model);
                boundModelHandle = draw.model;
                haveBoundModel = true;
                if (!boundModel)
                {
                    continue;
                }

                D3D12_VERTEX_BUFFER_VIEW vbv{};
                vbv.BufferLocation = boundModel->vertexBuffer.gpuAddress;
                vbv.SizeInBytes =
                    static_cast<UINT>(boundModel->vertexBuffer.size);
                vbv.StrideInBytes = sizeof(AssetVertex);

                D3D12_INDEX_BUFFER_VIEW ibv{};
                ibv.BufferLocation = boundModel->indexBuffer.gpuAddress;
                ibv.SizeInBytes =
                    static_cast<UINT>(boundModel->indexBuffer.size);
                ibv.Format = DXGI_FORMAT_R32_UINT;

                cmd->IASetVertexBuffers(0, 1, &vbv);
                cmd->IASetIndexBuffer(&ibv);
            }

            if (!boundModel || draw.meshIndex >= boundModel->meshes.size())
            {
                continue;
            }

            const GpuMesh& mesh = boundModel->meshes[draw.meshIndex];
            if (mesh.indexCount == 0)
            {
                continue;
            }

            DrawConstants constants{};
            constants.objectIndex = draw.objectIndex;
            constants.meshIndex = draw.meshIndex;
            constants.materialIndex = draw.materialIndex;

            const UINT drawRootParameter = 3u;
            cmd->SetGraphicsRoot32BitConstants(
                drawRootParameter,
                4,
                &constants,
                0);

            cmd->DrawIndexedInstanced(
                mesh.indexCount,
                1,
                mesh.firstIndex,
                0,
                0);
        }
    }

    void recordGpuFrustumCull(
        const DX12PassContext& ctx,
        const DX12CullBuffers& buffers)
    {
        ID3D12GraphicsCommandList4* cmd = ctx.cmd;

        cmd->SetComputeRootConstantBufferView(0, buffers.frameConstantsAddr);
        cmd->SetComputeRootShaderResourceView(1, buffers.instanceBoundsAddr);
        cmd->SetComputeRootUnorderedAccessView(2, buffers.visibleInstancesAddr);
        cmd->SetComputeRootUnorderedAccessView(
            3, buffers.visibleInstanceCountAddr);
        cmd->SetComputeRootShaderResourceView(4, buffers.drawInputsAddr);
        cmd->SetComputeRootUnorderedAccessView(
            5, buffers.indirectArgumentsAddr);
        cmd->SetComputeRootUnorderedAccessView(6, buffers.binCountsAddr);
    }
}
