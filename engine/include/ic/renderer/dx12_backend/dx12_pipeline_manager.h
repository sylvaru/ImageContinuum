#pragma once

#include <cstddef>
#include <vector>

#include <d3d12.h>
#include <wrl/client.h>

#include "ic/renderer/dx12_backend/dx12_device.h"
#include "ic/renderer/render_pipeline.h"

namespace ic
{
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

        static std::vector<std::byte> readBinaryFile(
            const std::filesystem::path& path);

        ID3D12Device5* m_device = nullptr;
        std::vector<DX12GraphicsPipeline> m_graphicsPipelines;
        std::vector<DX12ComputePipeline> m_computePipelines;
    };
}
