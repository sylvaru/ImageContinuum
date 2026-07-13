#include "ic/renderer/dx12_backend/dx12_pipeline_manager.h"

#include <cstddef>
#include <fstream>
#include <stdexcept>

#include <d3dcompiler.h>

#include "ic/core/asset_manager.h"
#include "ic/util/util.h"
#include "ic/renderer/renderer_common/renderer_util.h"

namespace
{
    void throwIfFailed(HRESULT hr, const char* message)
    {
        if (FAILED(hr))
        {
            throw std::runtime_error(message);
        }
    }
}

namespace ic
{
    void DX12PipelineManager::init(const DX12Device& device)
    {
        m_device = device.device();
        if (!m_device)
        {
            throw std::runtime_error(
                "DX12PipelineManager requires a valid device.");
        }
    }

    void DX12PipelineManager::shutdown()
    {
        m_graphicsPipelines.clear();
        m_computePipelines.clear();
        m_device = nullptr;
    }

    GraphicsPipelineHandle DX12PipelineManager::requestGraphicsPipeline(
        const GraphicsPipelineDesc& desc)
    {
        for (const DX12GraphicsPipeline& pipeline : m_graphicsPipelines)
        {
            if (pipeline.desc.debugName == desc.debugName)
            {
                return pipeline.handle;
            }
        }

        GraphicsPipelineHandle handle{};
        handle.index = static_cast<uint32_t>(m_graphicsPipelines.size());
        handle.generation = 1;

        m_graphicsPipelines.push_back(
            createGraphicsPipeline(handle, desc));

        return handle;
    }

    ComputePipelineHandle DX12PipelineManager::requestComputePipeline(
        const ComputePipelineDesc& desc)
    {
        for (const DX12ComputePipeline& pipeline : m_computePipelines)
        {
            if (pipeline.desc.debugName == desc.debugName)
            {
                return pipeline.handle;
            }
        }

        ComputePipelineHandle handle{};
        handle.index = static_cast<uint32_t>(m_computePipelines.size());
        handle.generation = 1;

        m_computePipelines.push_back(
            createComputePipeline(handle, desc));

        return handle;
    }

    DX12GraphicsPipeline* DX12PipelineManager::graphicsPipeline(
        GraphicsPipelineHandle handle)
    {
        return const_cast<DX12GraphicsPipeline*>(
            static_cast<const DX12PipelineManager&>(*this).graphicsPipeline(handle));
    }

    const DX12GraphicsPipeline* DX12PipelineManager::graphicsPipeline(
        GraphicsPipelineHandle handle) const
    {
        if (!handle || handle.index >= m_graphicsPipelines.size())
        {
            return nullptr;
        }

        const DX12GraphicsPipeline& pipeline =
            m_graphicsPipelines[handle.index];
        return pipeline.handle.generation == handle.generation
            ? &pipeline
            : nullptr;
    }

    DX12ComputePipeline* DX12PipelineManager::computePipeline(
        ComputePipelineHandle handle)
    {
        return const_cast<DX12ComputePipeline*>(
            static_cast<const DX12PipelineManager&>(*this).computePipeline(handle));
    }

    const DX12ComputePipeline* DX12PipelineManager::computePipeline(
        ComputePipelineHandle handle) const
    {
        if (!handle || handle.index >= m_computePipelines.size())
        {
            return nullptr;
        }

        const DX12ComputePipeline& pipeline =
            m_computePipelines[handle.index];
        return pipeline.handle.generation == handle.generation
            ? &pipeline
            : nullptr;
    }

    DX12GraphicsPipeline DX12PipelineManager::createGraphicsPipeline(
        GraphicsPipelineHandle handle,
        const GraphicsPipelineDesc& desc) const
    {
        if (!m_device)
        {
            throw std::runtime_error("DX12PipelineManager is not initialized.");
        }

        const std::vector<std::byte> vs =
            readBinaryFile(desc.shaders.vertexShader);
        const std::vector<std::byte> ps =
            readBinaryFile(desc.shaders.pixelShader);

        DX12GraphicsPipeline pipeline{};
        pipeline.handle = handle;
        pipeline.desc = desc;
        pipeline.rootSignature =
            createRootSignature(desc.bindingLayout);

        const std::vector<D3D12_INPUT_ELEMENT_DESC> inputElements =
            createInputLayout(desc.vertexLayout);

        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = pipeline.rootSignature.Get();
        pso.VS = { vs.data(), vs.size() };
        pso.PS = { ps.data(), ps.size() };
        pso.BlendState.RenderTarget[0].RenderTargetWriteMask =
            D3D12_COLOR_WRITE_ENABLE_ALL;
        pso.SampleMask = UINT_MAX;
        pso.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pso.RasterizerState.CullMode =
            toDx12CullMode(desc.raster.cullMode);
        pso.RasterizerState.DepthClipEnable =
            desc.raster.depthClip ? TRUE : FALSE;

        pso.DepthStencilState.DepthEnable =
            desc.depth.enabled ? TRUE : FALSE;
        pso.DepthStencilState.DepthWriteMask =
            desc.depth.write
                ? D3D12_DEPTH_WRITE_MASK_ALL
                : D3D12_DEPTH_WRITE_MASK_ZERO;
        pso.DepthStencilState.DepthFunc =
            toDx12CompareOp(desc.depth.compare);
        pso.DepthStencilState.StencilEnable = FALSE;

        pso.InputLayout = {
            inputElements.data(),
            static_cast<UINT>(inputElements.size())
        };
        pso.PrimitiveTopologyType = toDx12Topology(desc.topology);
        pso.NumRenderTargets = desc.colorAttachmentCount;

        for (uint32_t i = 0; i < desc.colorAttachmentCount; ++i)
        {
            pso.RTVFormats[i] = toDxgiFormat(desc.colorFormats[i]);
        }

        pso.DSVFormat = toDxgiFormat(desc.depth.format);
        pso.SampleDesc.Count = 1;

        throwIfFailed(
            m_device->CreateGraphicsPipelineState(
                &pso,
                IID_PPV_ARGS(&pipeline.pipelineState)),
            "Failed to create DX12 graphics pipeline state.");

        return pipeline;
    }

    DX12ComputePipeline DX12PipelineManager::createComputePipeline(
        ComputePipelineHandle handle,
        const ComputePipelineDesc& desc) const
    {
        if (!m_device)
        {
            throw std::runtime_error("DX12PipelineManager is not initialized.");
        }

        const std::vector<std::byte> cs =
            readBinaryFile(desc.shaders.computeShader);

        DX12ComputePipeline pipeline{};
        pipeline.handle = handle;
        pipeline.desc = desc;
        pipeline.rootSignature =
            createRootSignature(desc.bindingLayout);

        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = pipeline.rootSignature.Get();
        pso.CS = { cs.data(), cs.size() };

        throwIfFailed(
            m_device->CreateComputePipelineState(
                &pso,
                IID_PPV_ARGS(&pipeline.pipelineState)),
            "Failed to create DX12 compute pipeline state.");

        return pipeline;
    }

    Microsoft::WRL::ComPtr<ID3D12RootSignature>
        DX12PipelineManager::createRootSignature(
            PipelineBindingLayoutKind layout) const
    {
        if (layout == PipelineBindingLayoutKind::Empty)
        {
            D3D12_ROOT_SIGNATURE_DESC rootDesc{};
            rootDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

            Microsoft::WRL::ComPtr<ID3DBlob> signature;
            Microsoft::WRL::ComPtr<ID3DBlob> error;
            HRESULT hr = D3D12SerializeRootSignature(
                &rootDesc,
                D3D_ROOT_SIGNATURE_VERSION_1,
                &signature,
                &error);

            if (FAILED(hr))
            {
                const char* message = error
                    ? static_cast<const char*>(error->GetBufferPointer())
                    : "Failed to serialize empty DX12 root signature.";
                throw std::runtime_error(message);
            }

            Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSignature;
            throwIfFailed(
                m_device->CreateRootSignature(
                    0,
                    signature->GetBufferPointer(),
                    signature->GetBufferSize(),
                    IID_PPV_ARGS(&rootSignature)),
                "Failed to create empty DX12 root signature.");

            return rootSignature;
        }

        if (layout == PipelineBindingLayoutKind::ComputeStorageBuffer)
        {
            D3D12_ROOT_PARAMETER rootParameter{};
            rootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            rootParameter.Descriptor.ShaderRegister = 0;
            rootParameter.Descriptor.RegisterSpace = 0;
            rootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            D3D12_ROOT_SIGNATURE_DESC rootDesc{};
            rootDesc.NumParameters = 1;
            rootDesc.pParameters = &rootParameter;
            rootDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

            Microsoft::WRL::ComPtr<ID3DBlob> signature;
            Microsoft::WRL::ComPtr<ID3DBlob> error;
            HRESULT hr = D3D12SerializeRootSignature(
                &rootDesc,
                D3D_ROOT_SIGNATURE_VERSION_1,
                &signature,
                &error);

            if (FAILED(hr))
            {
                const char* message = error
                    ? static_cast<const char*>(error->GetBufferPointer())
                    : "Failed to serialize DX12 compute root signature.";
                throw std::runtime_error(message);
            }

            Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSignature;
            throwIfFailed(
                m_device->CreateRootSignature(
                    0,
                    signature->GetBufferPointer(),
                    signature->GetBufferSize(),
                    IID_PPV_ARGS(&rootSignature)),
                "Failed to create DX12 compute root signature.");

            return rootSignature;
        }

        if (layout == PipelineBindingLayoutKind::PathTrace ||
            layout == PipelineBindingLayoutKind::PathTraceTonemap)
        {
            D3D12_ROOT_PARAMETER rootParameters[6]{};

            rootParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
            rootParameters[0].Descriptor.ShaderRegister = 0;
            rootParameters[0].Descriptor.RegisterSpace = 0;
            rootParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            D3D12_DESCRIPTOR_RANGE uavRange{};
            uavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            uavRange.NumDescriptors = 1;
            uavRange.BaseShaderRegister = 1;
            uavRange.RegisterSpace = 0;
            uavRange.OffsetInDescriptorsFromTableStart = 0;

            rootParameters[1].ParameterType =
                D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            rootParameters[1].DescriptorTable.NumDescriptorRanges = 1;
            rootParameters[1].DescriptorTable.pDescriptorRanges = &uavRange;
            rootParameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            D3D12_DESCRIPTOR_RANGE srvRange{};
            srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            srvRange.NumDescriptors =
                layout == PipelineBindingLayoutKind::PathTrace
                    ? 5u
                    : 1u;
            srvRange.BaseShaderRegister =
                layout == PipelineBindingLayoutKind::PathTrace
                    ? 2u
                    : 2u;
            srvRange.RegisterSpace = 0;
            srvRange.OffsetInDescriptorsFromTableStart = 0;

            rootParameters[2].ParameterType =
                D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            rootParameters[2].DescriptorTable.NumDescriptorRanges = 1;
            rootParameters[2].DescriptorTable.pDescriptorRanges = &srvRange;
            rootParameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            D3D12_DESCRIPTOR_RANGE samplerRange{};
            samplerRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
            samplerRange.NumDescriptors = 1;
            samplerRange.BaseShaderRegister = 7;
            samplerRange.RegisterSpace = 0;
            samplerRange.OffsetInDescriptorsFromTableStart = 0;

            rootParameters[3].ParameterType =
                D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            rootParameters[3].DescriptorTable.NumDescriptorRanges = 1;
            rootParameters[3].DescriptorTable.pDescriptorRanges = &samplerRange;
            rootParameters[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            D3D12_DESCRIPTOR_RANGE bindlessTextureRange{};
            bindlessTextureRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            bindlessTextureRange.NumDescriptors = MaxBindlessTextures;
            bindlessTextureRange.BaseShaderRegister = MaxBindlessTextures + 2;
            bindlessTextureRange.RegisterSpace = 0;
            bindlessTextureRange.OffsetInDescriptorsFromTableStart = 0;

            rootParameters[4].ParameterType =
                D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            rootParameters[4].DescriptorTable.NumDescriptorRanges = 1;
            rootParameters[4].DescriptorTable.pDescriptorRanges =
                &bindlessTextureRange;
            rootParameters[4].ShaderVisibility =
                D3D12_SHADER_VISIBILITY_ALL;

            D3D12_DESCRIPTOR_RANGE bindlessSamplerRange{};
            bindlessSamplerRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
            bindlessSamplerRange.NumDescriptors = MaxBindlessSamplers;
            bindlessSamplerRange.BaseShaderRegister = MaxBindlessSamplers;
            bindlessSamplerRange.RegisterSpace = 0;
            bindlessSamplerRange.OffsetInDescriptorsFromTableStart = 0;

            rootParameters[5].ParameterType =
                D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            rootParameters[5].DescriptorTable.NumDescriptorRanges = 1;
            rootParameters[5].DescriptorTable.pDescriptorRanges =
                &bindlessSamplerRange;
            rootParameters[5].ShaderVisibility =
                D3D12_SHADER_VISIBILITY_ALL;

            D3D12_ROOT_SIGNATURE_DESC rootDesc{};
            rootDesc.NumParameters =
                layout == PipelineBindingLayoutKind::PathTrace ? 6u : 3u;
            rootDesc.pParameters = rootParameters;
            rootDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

            Microsoft::WRL::ComPtr<ID3DBlob> signature;
            Microsoft::WRL::ComPtr<ID3DBlob> error;
            HRESULT hr = D3D12SerializeRootSignature(
                &rootDesc,
                D3D_ROOT_SIGNATURE_VERSION_1,
                &signature,
                &error);

            if (FAILED(hr))
            {
                const char* message = error
                    ? static_cast<const char*>(error->GetBufferPointer())
                    : "Failed to serialize DX12 path tracing root signature.";
                throw std::runtime_error(message);
            }

            Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSignature;
            throwIfFailed(
                m_device->CreateRootSignature(
                    0,
                    signature->GetBufferPointer(),
                    signature->GetBufferSize(),
                    IID_PPV_ARGS(&rootSignature)),
                "Failed to create DX12 path tracing root signature.");

            return rootSignature;
        }

        if (layout == PipelineBindingLayoutKind::EnvironmentConvert)
        {
            D3D12_ROOT_PARAMETER rootParameters[3]{};
            D3D12_DESCRIPTOR_RANGE ranges[3]{};

            ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[0].NumDescriptors = 1;
            ranges[0].BaseShaderRegister = 0;
            ranges[0].RegisterSpace = 0;
            ranges[0].OffsetInDescriptorsFromTableStart = 0;

            ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            ranges[1].NumDescriptors = 1;
            ranges[1].BaseShaderRegister = 1;
            ranges[1].RegisterSpace = 0;
            ranges[1].OffsetInDescriptorsFromTableStart = 0;

            ranges[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
            ranges[2].NumDescriptors = 1;
            ranges[2].BaseShaderRegister = 2;
            ranges[2].RegisterSpace = 0;
            ranges[2].OffsetInDescriptorsFromTableStart = 0;

            for (UINT i = 0; i < static_cast<UINT>(std::size(rootParameters)); ++i)
            {
                rootParameters[i].ParameterType =
                    D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                rootParameters[i].DescriptorTable.NumDescriptorRanges = 1;
                rootParameters[i].DescriptorTable.pDescriptorRanges = &ranges[i];
                rootParameters[i].ShaderVisibility =
                    D3D12_SHADER_VISIBILITY_ALL;
            }

            D3D12_ROOT_SIGNATURE_DESC rootDesc{};
            rootDesc.NumParameters =
                static_cast<UINT>(std::size(rootParameters));
            rootDesc.pParameters = rootParameters;
            rootDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

            Microsoft::WRL::ComPtr<ID3DBlob> signature;
            Microsoft::WRL::ComPtr<ID3DBlob> error;
            HRESULT hr = D3D12SerializeRootSignature(
                &rootDesc,
                D3D_ROOT_SIGNATURE_VERSION_1,
                &signature,
                &error);

            if (FAILED(hr))
            {
                const char* message = error
                    ? static_cast<const char*>(error->GetBufferPointer())
                    : "Failed to serialize DX12 environment root signature.";
                throw std::runtime_error(message);
            }

            Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSignature;
            throwIfFailed(
                m_device->CreateRootSignature(
                    0,
                    signature->GetBufferPointer(),
                    signature->GetBufferSize(),
                    IID_PPV_ARGS(&rootSignature)),
                "Failed to create DX12 environment root signature.");

            return rootSignature;
        }

        if (layout == PipelineBindingLayoutKind::Skybox)
        {
            D3D12_ROOT_PARAMETER rootParameters[3]{};

            rootParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
            rootParameters[0].Descriptor.ShaderRegister = 0;
            rootParameters[0].Descriptor.RegisterSpace = 0;
            rootParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            D3D12_DESCRIPTOR_RANGE cubemapRange{};
            cubemapRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            cubemapRange.NumDescriptors = 1;
            cubemapRange.BaseShaderRegister = 0;
            cubemapRange.RegisterSpace = 0;
            cubemapRange.OffsetInDescriptorsFromTableStart = 0;

            rootParameters[1].ParameterType =
                D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            rootParameters[1].DescriptorTable.NumDescriptorRanges = 1;
            rootParameters[1].DescriptorTable.pDescriptorRanges = &cubemapRange;
            rootParameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            D3D12_DESCRIPTOR_RANGE samplerRange{};
            samplerRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
            samplerRange.NumDescriptors = 1;
            samplerRange.BaseShaderRegister = 0;
            samplerRange.RegisterSpace = 0;
            samplerRange.OffsetInDescriptorsFromTableStart = 0;

            rootParameters[2].ParameterType =
                D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            rootParameters[2].DescriptorTable.NumDescriptorRanges = 1;
            rootParameters[2].DescriptorTable.pDescriptorRanges = &samplerRange;
            rootParameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            D3D12_ROOT_SIGNATURE_DESC rootDesc{};
            rootDesc.NumParameters =
                static_cast<UINT>(std::size(rootParameters));
            rootDesc.pParameters = rootParameters;
            rootDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

            Microsoft::WRL::ComPtr<ID3DBlob> signature;
            Microsoft::WRL::ComPtr<ID3DBlob> error;
            HRESULT hr = D3D12SerializeRootSignature(
                &rootDesc,
                D3D_ROOT_SIGNATURE_VERSION_1,
                &signature,
                &error);

            if (FAILED(hr))
            {
                const char* message = error
                    ? static_cast<const char*>(error->GetBufferPointer())
                    : "Failed to serialize DX12 skybox root signature.";
                throw std::runtime_error(message);
            }

            Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSignature;
            throwIfFailed(
                m_device->CreateRootSignature(
                    0,
                    signature->GetBufferPointer(),
                    signature->GetBufferSize(),
                    IID_PPV_ARGS(&rootSignature)),
                "Failed to create DX12 skybox root signature.");

            return rootSignature;
        }

        if (layout == PipelineBindingLayoutKind::HiZDepthPyramid)
        {
            D3D12_ROOT_PARAMETER rootParameters[3]{};

            rootParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
            rootParameters[0].Descriptor.ShaderRegister = 0;
            rootParameters[0].Descriptor.RegisterSpace = 0;
            rootParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            D3D12_DESCRIPTOR_RANGE sourceRange{};
            sourceRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            sourceRange.NumDescriptors = 1;
            sourceRange.BaseShaderRegister = 20;
            sourceRange.RegisterSpace = 0;
            sourceRange.OffsetInDescriptorsFromTableStart = 0;

            rootParameters[1].ParameterType =
                D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            rootParameters[1].DescriptorTable.NumDescriptorRanges = 1;
            rootParameters[1].DescriptorTable.pDescriptorRanges = &sourceRange;
            rootParameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            D3D12_DESCRIPTOR_RANGE outputRange{};
            outputRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            outputRange.NumDescriptors = 1;
            outputRange.BaseShaderRegister = 21;
            outputRange.RegisterSpace = 0;
            outputRange.OffsetInDescriptorsFromTableStart = 0;

            rootParameters[2].ParameterType =
                D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            rootParameters[2].DescriptorTable.NumDescriptorRanges = 1;
            rootParameters[2].DescriptorTable.pDescriptorRanges = &outputRange;
            rootParameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            D3D12_ROOT_SIGNATURE_DESC rootDesc{};
            rootDesc.NumParameters =
                static_cast<UINT>(std::size(rootParameters));
            rootDesc.pParameters = rootParameters;

            Microsoft::WRL::ComPtr<ID3DBlob> signature;
            Microsoft::WRL::ComPtr<ID3DBlob> error;
            HRESULT hr = D3D12SerializeRootSignature(
                &rootDesc,
                D3D_ROOT_SIGNATURE_VERSION_1,
                &signature,
                &error);
            if (FAILED(hr))
            {
                const char* message = error
                    ? static_cast<const char*>(error->GetBufferPointer())
                    : "Failed to serialize DX12 Hi-Z root signature.";
                throw std::runtime_error(message);
            }

            Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSignature;
            throwIfFailed(
                m_device->CreateRootSignature(
                    0,
                    signature->GetBufferPointer(),
                    signature->GetBufferSize(),
                    IID_PPV_ARGS(&rootSignature)),
                "Failed to create DX12 Hi-Z root signature.");
            return rootSignature;
        }

        if (layout == PipelineBindingLayoutKind::GpuFrustumCull)
        {
            D3D12_ROOT_PARAMETER rootParameters[7]{};
            rootParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
            rootParameters[0].Descriptor.ShaderRegister = 0;
            rootParameters[0].Descriptor.RegisterSpace = 0;
            rootParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            rootParameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            rootParameters[1].Descriptor.ShaderRegister = 22;
            rootParameters[1].Descriptor.RegisterSpace = 0;
            rootParameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            rootParameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            rootParameters[2].Descriptor.ShaderRegister = 23;
            rootParameters[2].Descriptor.RegisterSpace = 0;
            rootParameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            rootParameters[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            rootParameters[3].Descriptor.ShaderRegister = 24;
            rootParameters[3].Descriptor.RegisterSpace = 0;
            rootParameters[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            const UINT registers[] = { 25u, 26u, 28u };
            const D3D12_ROOT_PARAMETER_TYPE types[] = {
                D3D12_ROOT_PARAMETER_TYPE_SRV,
                D3D12_ROOT_PARAMETER_TYPE_UAV,
                D3D12_ROOT_PARAMETER_TYPE_UAV };
            for (UINT i = 0; i < 3u; ++i)
            {
                rootParameters[4u + i].ParameterType = types[i];
                rootParameters[4u + i].Descriptor.ShaderRegister = registers[i];
                rootParameters[4u + i].Descriptor.RegisterSpace = 0;
                rootParameters[4u + i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            }

            D3D12_ROOT_SIGNATURE_DESC rootDesc{};
            rootDesc.NumParameters =
                static_cast<UINT>(std::size(rootParameters));
            rootDesc.pParameters = rootParameters;

            Microsoft::WRL::ComPtr<ID3DBlob> signature;
            Microsoft::WRL::ComPtr<ID3DBlob> error;
            HRESULT hr = D3D12SerializeRootSignature(
                &rootDesc,
                D3D_ROOT_SIGNATURE_VERSION_1,
                &signature,
                &error);
            if (FAILED(hr))
            {
                const char* message = error
                    ? static_cast<const char*>(error->GetBufferPointer())
                    : "Failed to serialize DX12 GPU frustum cull root signature.";
                throw std::runtime_error(message);
            }

            Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSignature;
            throwIfFailed(
                m_device->CreateRootSignature(
                    0,
                    signature->GetBufferPointer(),
                    signature->GetBufferSize(),
                    IID_PPV_ARGS(&rootSignature)),
                "Failed to create DX12 GPU frustum cull root signature.");
            return rootSignature;
        }

        if (layout == PipelineBindingLayoutKind::ClusteredForward)
        {
            D3D12_ROOT_PARAMETER rootParameters[18]{};

            // 0: Frame constants, b0 space0
            rootParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
            rootParameters[0].Descriptor.ShaderRegister = 0;
            rootParameters[0].Descriptor.RegisterSpace = 0;
            rootParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            // 1: Objects, t0 space0
            D3D12_DESCRIPTOR_RANGE objectRange{};
            objectRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            objectRange.NumDescriptors = 1;
            objectRange.BaseShaderRegister = 0;
            objectRange.RegisterSpace = 0;
            objectRange.OffsetInDescriptorsFromTableStart = 0;

            rootParameters[1].ParameterType =
                D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            rootParameters[1].DescriptorTable.NumDescriptorRanges = 1;
            rootParameters[1].DescriptorTable.pDescriptorRanges = &objectRange;
            rootParameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            // 2: Materials, t1 space0
            D3D12_DESCRIPTOR_RANGE materialRange{};
            materialRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            materialRange.NumDescriptors = 1;
            materialRange.BaseShaderRegister = 1;
            materialRange.RegisterSpace = 0;
            materialRange.OffsetInDescriptorsFromTableStart = 0;

            rootParameters[2].ParameterType =
                D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            rootParameters[2].DescriptorTable.NumDescriptorRanges = 1;
            rootParameters[2].DescriptorTable.pDescriptorRanges = &materialRange;
            rootParameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            // 3: Draw constants, b1/root constants space0
            rootParameters[3].ParameterType =
                D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            rootParameters[3].Constants.ShaderRegister = 1;
            rootParameters[3].Constants.RegisterSpace = 0;
            rootParameters[3].Constants.Num32BitValues = 4;
            rootParameters[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            // 4: Bindless textures, t2+ space0
            D3D12_DESCRIPTOR_RANGE bindlessTextureRange{};
            bindlessTextureRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            bindlessTextureRange.NumDescriptors = MaxBindlessTextures;
            bindlessTextureRange.BaseShaderRegister = 2;
            bindlessTextureRange.RegisterSpace = 0;
            bindlessTextureRange.OffsetInDescriptorsFromTableStart = 0;

            rootParameters[4].ParameterType =
                D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            rootParameters[4].DescriptorTable.NumDescriptorRanges = 1;
            rootParameters[4].DescriptorTable.pDescriptorRanges =
                &bindlessTextureRange;
            rootParameters[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

            // 5: Bindless samplers, s0+ space0
            D3D12_DESCRIPTOR_RANGE bindlessSamplerRange{};
            bindlessSamplerRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
            bindlessSamplerRange.NumDescriptors = MaxBindlessSamplers;
            bindlessSamplerRange.BaseShaderRegister = 0;
            bindlessSamplerRange.RegisterSpace = 0;
            bindlessSamplerRange.OffsetInDescriptorsFromTableStart = 0;

            rootParameters[5].ParameterType =
                D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            rootParameters[5].DescriptorTable.NumDescriptorRanges = 1;
            rootParameters[5].DescriptorTable.pDescriptorRanges =
                &bindlessSamplerRange;
            rootParameters[5].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

            // 6: IBL irradiance, t0 space1
            D3D12_DESCRIPTOR_RANGE irradianceRange{};
            irradianceRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            irradianceRange.NumDescriptors = 1;
            irradianceRange.BaseShaderRegister = 0;
            irradianceRange.RegisterSpace = 1;
            irradianceRange.OffsetInDescriptorsFromTableStart = 0;

            rootParameters[6].ParameterType =
                D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            rootParameters[6].DescriptorTable.NumDescriptorRanges = 1;
            rootParameters[6].DescriptorTable.pDescriptorRanges =
                &irradianceRange;
            rootParameters[6].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

            // 7: IBL prefiltered environment, t1 space1
            D3D12_DESCRIPTOR_RANGE prefilteredRange{};
            prefilteredRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            prefilteredRange.NumDescriptors = 1;
            prefilteredRange.BaseShaderRegister = 1;
            prefilteredRange.RegisterSpace = 1;
            prefilteredRange.OffsetInDescriptorsFromTableStart = 0;

            rootParameters[7].ParameterType =
                D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            rootParameters[7].DescriptorTable.NumDescriptorRanges = 1;
            rootParameters[7].DescriptorTable.pDescriptorRanges =
                &prefilteredRange;
            rootParameters[7].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

            // 8: BRDF LUT, t2 space1
            D3D12_DESCRIPTOR_RANGE brdfLutRange{};
            brdfLutRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            brdfLutRange.NumDescriptors = 1;
            brdfLutRange.BaseShaderRegister = 2;
            brdfLutRange.RegisterSpace = 1;
            brdfLutRange.OffsetInDescriptorsFromTableStart = 0;

            rootParameters[8].ParameterType =
                D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            rootParameters[8].DescriptorTable.NumDescriptorRanges = 1;
            rootParameters[8].DescriptorTable.pDescriptorRanges =
                &brdfLutRange;
            rootParameters[8].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

            // 9: IBL sampler, s0 space1
            D3D12_DESCRIPTOR_RANGE iblSamplerRange{};
            iblSamplerRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
            iblSamplerRange.NumDescriptors = 1;
            iblSamplerRange.BaseShaderRegister = 0;
            iblSamplerRange.RegisterSpace = 1;
            iblSamplerRange.OffsetInDescriptorsFromTableStart = 0;

            rootParameters[9].ParameterType =
                D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            rootParameters[9].DescriptorTable.NumDescriptorRanges = 1;
            rootParameters[9].DescriptorTable.pDescriptorRanges =
                &iblSamplerRange;
            rootParameters[9].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

            // 10: Cluster bounds UAV, u10 space2
            rootParameters[10].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            rootParameters[10].Descriptor.ShaderRegister = 10;
            rootParameters[10].Descriptor.RegisterSpace = 2;
            rootParameters[10].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            // 11: Cluster light grid UAV, u11 space2
            rootParameters[11].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            rootParameters[11].Descriptor.ShaderRegister = 11;
            rootParameters[11].Descriptor.RegisterSpace = 2;
            rootParameters[11].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            // 12: Cluster light indices UAV, u12 space2
            rootParameters[12].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            rootParameters[12].Descriptor.ShaderRegister = 12;
            rootParameters[12].Descriptor.RegisterSpace = 2;
            rootParameters[12].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            // 13: Visible lights SRV, t13 space2
            rootParameters[13].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            rootParameters[13].Descriptor.ShaderRegister = 13;
            rootParameters[13].Descriptor.RegisterSpace = 2;
            rootParameters[13].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            // 14: Cluster light counter UAV, u14 space2
            rootParameters[14].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            rootParameters[14].Descriptor.ShaderRegister = 14;
            rootParameters[14].Descriptor.RegisterSpace = 2;
            rootParameters[14].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            // 15: Cluster bounds SRV, t10 space2
            rootParameters[15].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            rootParameters[15].Descriptor.ShaderRegister = 10;
            rootParameters[15].Descriptor.RegisterSpace = 2;
            rootParameters[15].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            // 16: Cluster light grid SRV, t11 space2
            rootParameters[16].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            rootParameters[16].Descriptor.ShaderRegister = 11;
            rootParameters[16].Descriptor.RegisterSpace = 2;
            rootParameters[16].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            // 17: Cluster light indices SRV, t12 space2
            rootParameters[17].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            rootParameters[17].Descriptor.ShaderRegister = 12;
            rootParameters[17].Descriptor.RegisterSpace = 2;
            rootParameters[17].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            D3D12_ROOT_SIGNATURE_DESC rootDesc{};
            rootDesc.NumParameters =
                static_cast<UINT>(std::size(rootParameters));
            rootDesc.pParameters = rootParameters;
            rootDesc.Flags =
                D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

            Microsoft::WRL::ComPtr<ID3DBlob> signature;
            Microsoft::WRL::ComPtr<ID3DBlob> error;
            HRESULT hr = D3D12SerializeRootSignature(
                &rootDesc,
                D3D_ROOT_SIGNATURE_VERSION_1,
                &signature,
                &error);

            if (FAILED(hr))
            {
                const char* message = error
                    ? static_cast<const char*>(error->GetBufferPointer())
                    : "Failed to serialize DX12 clustered forward root signature.";
                throw std::runtime_error(message);
            }

            Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSignature;
            throwIfFailed(
                m_device->CreateRootSignature(
                    0,
                    signature->GetBufferPointer(),
                    signature->GetBufferSize(),
                    IID_PPV_ARGS(&rootSignature)),
                "Failed to create DX12 clustered forward root signature.");

            return rootSignature;
        }

        if (layout != PipelineBindingLayoutKind::ForwardBindless)
        {
            throw std::runtime_error(
                "Unsupported DX12 pipeline binding layout.");
        }

        D3D12_ROOT_PARAMETER rootParameters[10]{};

        rootParameters[0].ParameterType =
            D3D12_ROOT_PARAMETER_TYPE_CBV;
        rootParameters[0].Descriptor.ShaderRegister = 0;
        rootParameters[0].Descriptor.RegisterSpace = 0;
        rootParameters[0].ShaderVisibility =
            D3D12_SHADER_VISIBILITY_ALL;

        D3D12_DESCRIPTOR_RANGE objectRange{};
        objectRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        objectRange.NumDescriptors = 1;
        objectRange.BaseShaderRegister = 0;
        objectRange.RegisterSpace = 0;
        objectRange.OffsetInDescriptorsFromTableStart = 0;

        rootParameters[1].ParameterType =
            D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParameters[1].DescriptorTable.NumDescriptorRanges = 1;
        rootParameters[1].DescriptorTable.pDescriptorRanges = &objectRange;
        rootParameters[1].ShaderVisibility =
            D3D12_SHADER_VISIBILITY_ALL;

        D3D12_DESCRIPTOR_RANGE materialRange{};
        materialRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        materialRange.NumDescriptors = 1;
        materialRange.BaseShaderRegister = 1;
        materialRange.RegisterSpace = 0;
        materialRange.OffsetInDescriptorsFromTableStart = 0;

        rootParameters[2].ParameterType =
            D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParameters[2].DescriptorTable.NumDescriptorRanges = 1;
        rootParameters[2].DescriptorTable.pDescriptorRanges = &materialRange;
        rootParameters[2].ShaderVisibility =
            D3D12_SHADER_VISIBILITY_ALL;

        rootParameters[3].ParameterType =
            D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        rootParameters[3].Constants.ShaderRegister = 1;
        rootParameters[3].Constants.RegisterSpace = 0;
        rootParameters[3].Constants.Num32BitValues = 4;
        rootParameters[3].ShaderVisibility =
            D3D12_SHADER_VISIBILITY_ALL;

        D3D12_DESCRIPTOR_RANGE bindlessTextureRange{};
        bindlessTextureRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        bindlessTextureRange.NumDescriptors = MaxBindlessTextures;
        bindlessTextureRange.BaseShaderRegister = 2;
        bindlessTextureRange.RegisterSpace = 0;
        bindlessTextureRange.OffsetInDescriptorsFromTableStart = 0;

        rootParameters[4].ParameterType =
            D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParameters[4].DescriptorTable.NumDescriptorRanges = 1;
        rootParameters[4].DescriptorTable.pDescriptorRanges =
            &bindlessTextureRange;
        rootParameters[4].ShaderVisibility =
            D3D12_SHADER_VISIBILITY_PIXEL;

        D3D12_DESCRIPTOR_RANGE bindlessSamplerRange{};
        bindlessSamplerRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
        bindlessSamplerRange.NumDescriptors = MaxBindlessSamplers;
        bindlessSamplerRange.BaseShaderRegister = 0;
        bindlessSamplerRange.RegisterSpace = 0;
        bindlessSamplerRange.OffsetInDescriptorsFromTableStart = 0;

        rootParameters[5].ParameterType =
            D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParameters[5].DescriptorTable.NumDescriptorRanges = 1;
        rootParameters[5].DescriptorTable.pDescriptorRanges =
            &bindlessSamplerRange;
        rootParameters[5].ShaderVisibility =
            D3D12_SHADER_VISIBILITY_PIXEL;

        D3D12_DESCRIPTOR_RANGE irradianceRange{};
        irradianceRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        irradianceRange.NumDescriptors = 1;
        irradianceRange.BaseShaderRegister = 0;
        irradianceRange.RegisterSpace = 1;
        irradianceRange.OffsetInDescriptorsFromTableStart = 0;

        rootParameters[6].ParameterType =
            D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParameters[6].DescriptorTable.NumDescriptorRanges = 1;
        rootParameters[6].DescriptorTable.pDescriptorRanges =
            &irradianceRange;
        rootParameters[6].ShaderVisibility =
            D3D12_SHADER_VISIBILITY_PIXEL;

        D3D12_DESCRIPTOR_RANGE prefilteredRange{};
        prefilteredRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        prefilteredRange.NumDescriptors = 1;
        prefilteredRange.BaseShaderRegister = 1;
        prefilteredRange.RegisterSpace = 1;
        prefilteredRange.OffsetInDescriptorsFromTableStart = 0;

        rootParameters[7].ParameterType =
            D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParameters[7].DescriptorTable.NumDescriptorRanges = 1;
        rootParameters[7].DescriptorTable.pDescriptorRanges =
            &prefilteredRange;
        rootParameters[7].ShaderVisibility =
            D3D12_SHADER_VISIBILITY_PIXEL;

        D3D12_DESCRIPTOR_RANGE brdfLutRange{};
        brdfLutRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        brdfLutRange.NumDescriptors = 1;
        brdfLutRange.BaseShaderRegister = 2;
        brdfLutRange.RegisterSpace = 1;
        brdfLutRange.OffsetInDescriptorsFromTableStart = 0;

        rootParameters[8].ParameterType =
            D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParameters[8].DescriptorTable.NumDescriptorRanges = 1;
        rootParameters[8].DescriptorTable.pDescriptorRanges =
            &brdfLutRange;
        rootParameters[8].ShaderVisibility =
            D3D12_SHADER_VISIBILITY_PIXEL;

        D3D12_DESCRIPTOR_RANGE iblSamplerRange{};
        iblSamplerRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
        iblSamplerRange.NumDescriptors = 1;
        iblSamplerRange.BaseShaderRegister = 0;
        iblSamplerRange.RegisterSpace = 1;
        iblSamplerRange.OffsetInDescriptorsFromTableStart = 0;

        rootParameters[9].ParameterType =
            D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParameters[9].DescriptorTable.NumDescriptorRanges = 1;
        rootParameters[9].DescriptorTable.pDescriptorRanges =
            &iblSamplerRange;
        rootParameters[9].ShaderVisibility =
            D3D12_SHADER_VISIBILITY_PIXEL;

        D3D12_ROOT_SIGNATURE_DESC rootDesc{};
        rootDesc.NumParameters =
            static_cast<UINT>(std::size(rootParameters));
        rootDesc.pParameters = rootParameters;
        rootDesc.Flags =
            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

        Microsoft::WRL::ComPtr<ID3DBlob> signature;
        Microsoft::WRL::ComPtr<ID3DBlob> error;
        HRESULT hr = D3D12SerializeRootSignature(
            &rootDesc,
            D3D_ROOT_SIGNATURE_VERSION_1,
            &signature,
            &error);

        if (FAILED(hr))
        {
            const char* message = error
                ? static_cast<const char*>(error->GetBufferPointer())
                : "Failed to serialize DX12 root signature.";
            throw std::runtime_error(message);
        }

        Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSignature;
        throwIfFailed(
            m_device->CreateRootSignature(
                0,
                signature->GetBufferPointer(),
                signature->GetBufferSize(),
                IID_PPV_ARGS(&rootSignature)),
            "Failed to create DX12 root signature.");

        return rootSignature;
    }

    std::vector<D3D12_INPUT_ELEMENT_DESC>
        DX12PipelineManager::createInputLayout(VertexLayoutKind layout) const
    {
        if (layout == VertexLayoutKind::Unknown)
        {
            return {};
        }

        if (layout != VertexLayoutKind::AssetVertex)
        {
            throw std::runtime_error(
                "Unsupported DX12 vertex layout.");
        }

        return {
            { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, offsetof(AssetVertex, position), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            { "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, offsetof(AssetVertex, normal), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            { "TANGENT", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, offsetof(AssetVertex, tangent), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, offsetof(AssetVertex, uv0), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            { "TEXCOORD", 1, DXGI_FORMAT_R32G32_FLOAT, 0, offsetof(AssetVertex, uv1), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            { "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, offsetof(AssetVertex, color), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        };
    }

    D3D12_PRIMITIVE_TOPOLOGY_TYPE DX12PipelineManager::toDx12Topology(
        PrimitiveTopologyKind topology) const
    {
        switch (topology)
        {
        case PrimitiveTopologyKind::TriangleList:
            return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        }

        return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    }

    D3D12_CULL_MODE DX12PipelineManager::toDx12CullMode(CullMode mode) const
    {
        switch (mode)
        {
        case CullMode::None:
            return D3D12_CULL_MODE_NONE;
        case CullMode::Front:
            return D3D12_CULL_MODE_FRONT;
        case CullMode::Back:
            return D3D12_CULL_MODE_BACK;
        }

        return D3D12_CULL_MODE_BACK;
    }

    D3D12_COMPARISON_FUNC DX12PipelineManager::toDx12CompareOp(
        CompareOp compare) const
    {
        switch (compare)
        {
        case CompareOp::Never:
            return D3D12_COMPARISON_FUNC_NEVER;
        case CompareOp::Less:
            return D3D12_COMPARISON_FUNC_LESS;
        case CompareOp::Equal:
            return D3D12_COMPARISON_FUNC_EQUAL;
        case CompareOp::LessEqual:
            return D3D12_COMPARISON_FUNC_LESS_EQUAL;
        case CompareOp::Greater:
            return D3D12_COMPARISON_FUNC_GREATER;
        case CompareOp::Always:
            return D3D12_COMPARISON_FUNC_ALWAYS;
        }

        return D3D12_COMPARISON_FUNC_LESS_EQUAL;
    }

    DXGI_FORMAT DX12PipelineManager::toDxgiFormat(TextureFormat format) const
    {
        switch (format)
        {
        case TextureFormat::RGBA8_UNorm:
            return DXGI_FORMAT_R8G8B8A8_UNORM;
        case TextureFormat::RGBA8_SRGB:
            return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        case TextureFormat::RGBA32_Float:
            return DXGI_FORMAT_R32G32B32A32_FLOAT;
        case TextureFormat::BGRA8_UNorm:
            return DXGI_FORMAT_B8G8R8A8_UNORM;
        case TextureFormat::BGRA8_SRGB:
            return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
        case TextureFormat::D32_Float:
            return DXGI_FORMAT_D32_FLOAT;
        case TextureFormat::R32_Float:
            return DXGI_FORMAT_R32_FLOAT;
        case TextureFormat::Unknown:
            break;
        }

        return DXGI_FORMAT_UNKNOWN;
    }

}
