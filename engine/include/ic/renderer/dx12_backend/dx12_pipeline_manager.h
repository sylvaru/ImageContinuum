#pragma once

#include <cstddef>
#include <vector>
#include <deque>
#include <unordered_map>

#include <d3d12.h>
#include <wrl/client.h>

#include "ic/renderer/dx12_backend/dx12_device.h"
#include "ic/renderer/render_pipeline.h"

namespace ic
{
    class PipelineLibrary;
    struct DX12GraphicsPipeline
    {
        GraphicsPipelineHandle handle = {};
        GraphicsPipelineDesc desc = {};

        Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSignature;
        Microsoft::WRL::ComPtr<ID3D12PipelineState> pipelineState;

        explicit operator bool() const
        {
            return rootSignature && pipelineState;
        }
    };

    struct DX12ComputePipeline
    {
        ComputePipelineHandle handle = {};
        ComputePipelineDesc desc = {};

        Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSignature;
        Microsoft::WRL::ComPtr<ID3D12PipelineState> pipelineState;

        explicit operator bool() const
        {
            return rootSignature && pipelineState;
        }
    };

    class DX12PipelineManager final
    {
    public:
        void init(const DX12Device& device);
        void shutdown();

        GraphicsPipelineHandle requestGraphicsPipeline(
            const GraphicsPipelineDesc& desc);
        ComputePipelineHandle requestComputePipeline(
            const ComputePipelineDesc& desc);
        GraphicsPipelineHandle resolveGraphicsPipeline(
            const PipelineLibrary& library,
            PipelineId id,
            TextureFormat swapchainFormat);
        ComputePipelineHandle resolveComputePipeline(
            const PipelineLibrary& library,
            PipelineId id);

        DX12GraphicsPipeline* graphicsPipeline(GraphicsPipelineHandle handle);
        const DX12GraphicsPipeline* graphicsPipeline(GraphicsPipelineHandle handle) const;
        DX12ComputePipeline* computePipeline(ComputePipelineHandle handle);
        const DX12ComputePipeline* computePipeline(ComputePipelineHandle handle) const;

    private:
        DX12GraphicsPipeline createGraphicsPipeline(
            GraphicsPipelineHandle handle,
            const GraphicsPipelineDesc& desc) const;

        DX12ComputePipeline createComputePipeline(
            ComputePipelineHandle handle,
            const ComputePipelineDesc& desc) const;

        Microsoft::WRL::ComPtr<ID3D12RootSignature> createRootSignature(
            PipelineBindingLayoutKind layout) const;

        std::vector<D3D12_INPUT_ELEMENT_DESC> createInputLayout(
            VertexLayoutKind layout) const;

        D3D12_PRIMITIVE_TOPOLOGY_TYPE toDx12Topology(
            PrimitiveTopologyKind topology) const;

        D3D12_CULL_MODE toDx12CullMode(CullMode mode) const;
        D3D12_COMPARISON_FUNC toDx12CompareOp(CompareOp compare) const;
        DXGI_FORMAT toDxgiFormat(TextureFormat format) const;

        ID3D12Device5* m_device = nullptr;
        // Pipeline pointers are handed to pass recording code. deque keeps
        // element addresses stable as lazily resolved pipelines are appended.
        std::deque<DX12GraphicsPipeline> m_graphicsPipelines;
        std::deque<DX12ComputePipeline> m_computePipelines;
        std::unordered_map<PipelineId, GraphicsPipelineHandle, PipelineIdHash>
            m_graphicsById;
        std::unordered_map<PipelineId, ComputePipelineHandle, PipelineIdHash>
            m_computeById;
    };
}
