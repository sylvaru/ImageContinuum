#include "ic/renderer/dx12_backend/dx12_device.h"

#include <spdlog/spdlog.h>

#include <stdexcept>

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
    void DX12Device::init(const DX12Adapter& adapter, bool enableValidation)
    {
        if (!adapter.adapter())
        {
            throw std::runtime_error("DX12 device creation requires a valid adapter.");
        }

        spdlog::info("[DX12Device] Creating device...");

        throwIfFailed(
            D3D12CreateDevice(
                adapter.adapter(),
                D3D_FEATURE_LEVEL_12_0,
                IID_PPV_ARGS(&m_device)),
            "Failed to create D3D12 device.");

        if (enableValidation)
        {
            Microsoft::WRL::ComPtr<ID3D12InfoQueue> infoQueue;
            if (SUCCEEDED(m_device.As(&infoQueue)))
            {
                infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);
                infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);
            }
        }

        queryFeatureSupport();

        m_graphicsQueue = createQueue(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            D3D12_COMMAND_QUEUE_PRIORITY_NORMAL,
            L"IC DX12 Graphics Queue");

        m_computeQueue = createQueue(
            D3D12_COMMAND_LIST_TYPE_COMPUTE,
            D3D12_COMMAND_QUEUE_PRIORITY_NORMAL,
            L"IC DX12 Compute Queue");

        m_copyQueue = createQueue(
            D3D12_COMMAND_LIST_TYPE_COPY,
            D3D12_COMMAND_QUEUE_PRIORITY_NORMAL,
            L"IC DX12 Copy Queue");

        m_rtvDescriptorSize =
            m_device->GetDescriptorHandleIncrementSize(
                D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

        spdlog::info(
            "[DX12Device] Created (feature level 12_0, RTV descriptor size={}, binding tier={}, shader model={:#x})",
            m_rtvDescriptorSize,
            static_cast<uint32_t>(m_features.resourceBindingTier),
            static_cast<uint32_t>(m_features.shaderModel));
    }

    void DX12Device::shutdown()
    {
        m_copyQueue.Reset();
        m_computeQueue.Reset();
        m_graphicsQueue.Reset();
        m_device.Reset();
        m_features = {};
        m_rtvDescriptorSize = 0;
        spdlog::info("[DX12Device] Shutdown");
    }

    Microsoft::WRL::ComPtr<ID3D12CommandQueue> DX12Device::createQueue(
        D3D12_COMMAND_LIST_TYPE type,
        D3D12_COMMAND_QUEUE_PRIORITY priority,
        const wchar_t* name) const
    {
        D3D12_COMMAND_QUEUE_DESC desc{};
        desc.Type = type;
        desc.Priority = static_cast<INT>(priority);
        desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        desc.NodeMask = 0;

        Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue;
        throwIfFailed(
            m_device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue)),
            "Failed to create D3D12 command queue.");

        queue->SetName(name);
        return queue;
    }

    void DX12Device::queryFeatureSupport()
    {
        D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
        throwIfFailed(
            m_device->CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS,
                &options,
                sizeof(options)),
            "Failed to query D3D12 feature options.");

        m_features.resourceBindingTier =
            options.ResourceBindingTier;

        D3D12_FEATURE_DATA_ROOT_SIGNATURE rootSignature{};
        rootSignature.HighestVersion =
            D3D_ROOT_SIGNATURE_VERSION_1_1;

        if (FAILED(m_device->CheckFeatureSupport(
                D3D12_FEATURE_ROOT_SIGNATURE,
                &rootSignature,
                sizeof(rootSignature))))
        {
            rootSignature.HighestVersion =
                D3D_ROOT_SIGNATURE_VERSION_1_0;
        }

        m_features.rootSignatureVersion =
            rootSignature.HighestVersion;

        D3D12_FEATURE_DATA_SHADER_MODEL shaderModel{};
        shaderModel.HighestShaderModel =
            D3D_SHADER_MODEL_6_6;

        if (FAILED(m_device->CheckFeatureSupport(
                D3D12_FEATURE_SHADER_MODEL,
                &shaderModel,
                sizeof(shaderModel))))
        {
            shaderModel.HighestShaderModel =
                D3D_SHADER_MODEL_5_1;
        }

        m_features.shaderModel =
            shaderModel.HighestShaderModel;

        m_features.gpuVirtualAddress = true;
        m_features.descriptorIndexing =
            m_features.resourceBindingTier >= D3D12_RESOURCE_BINDING_TIER_2;

        m_features.bindlessResources =
            m_features.resourceBindingTier >= D3D12_RESOURCE_BINDING_TIER_3;

        m_features.directHeapIndexing =
            m_features.bindlessResources &&
            m_features.shaderModel >= D3D_SHADER_MODEL_6_6;

        if (!m_features.descriptorIndexing)
        {
            throw std::runtime_error(
                "Selected DX12 device does not support required descriptor indexing tier.");
        }

        spdlog::info(
            "[DX12Device] Features | resourceBindingTier={} rootSignature={} shaderModel={:#x} gpuVA={} bindless={} directHeapIndexing={}",
            static_cast<uint32_t>(m_features.resourceBindingTier),
            static_cast<uint32_t>(m_features.rootSignatureVersion),
            static_cast<uint32_t>(m_features.shaderModel),
            m_features.gpuVirtualAddress,
            m_features.bindlessResources,
            m_features.directHeapIndexing);
    }
}
