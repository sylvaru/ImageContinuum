#pragma once

#include "ic/renderer/dx12_backend/dx12_adapter.h"

#include <d3d12.h>
#include <wrl/client.h>

namespace ic
{
    struct DX12FeatureSupport
    {
        D3D12_RESOURCE_BINDING_TIER resourceBindingTier =
            D3D12_RESOURCE_BINDING_TIER_1;
        D3D_ROOT_SIGNATURE_VERSION rootSignatureVersion =
            D3D_ROOT_SIGNATURE_VERSION_1_0;
        D3D_SHADER_MODEL shaderModel =
            D3D_SHADER_MODEL_5_1;
        bool gpuVirtualAddress = false;
        bool descriptorIndexing = false;
        bool bindlessResources = false;
        bool directHeapIndexing = false;
    };

    class DX12Device
    {
    public:
        void init(const DX12Adapter& adapter, bool enableValidation);
        void shutdown();

        ID3D12Device5* device() const
        {
            return m_device.Get();
        }

        ID3D12CommandQueue* graphicsQueue() const
        {
            return m_graphicsQueue.Get();
        }

        ID3D12CommandQueue* computeQueue() const
        {
            return m_computeQueue.Get();
        }

        ID3D12CommandQueue* copyQueue() const
        {
            return m_copyQueue.Get();
        }

        uint32_t rtvDescriptorSize() const
        {
            return m_rtvDescriptorSize;
        }

        const DX12FeatureSupport& features() const
        {
            return m_features;
        }

        void logValidationMessages();

    private:
        void queryFeatureSupport();

        Microsoft::WRL::ComPtr<ID3D12CommandQueue> createQueue(
            D3D12_COMMAND_LIST_TYPE type,
            D3D12_COMMAND_QUEUE_PRIORITY priority,
            const wchar_t* name) const;

        Microsoft::WRL::ComPtr<ID3D12Device5> m_device;
        Microsoft::WRL::ComPtr<ID3D12InfoQueue> m_infoQueue;
        Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_graphicsQueue;
        Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_computeQueue;
        Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_copyQueue;
        DX12FeatureSupport m_features;
        uint32_t m_rtvDescriptorSize = 0;
    };
}
