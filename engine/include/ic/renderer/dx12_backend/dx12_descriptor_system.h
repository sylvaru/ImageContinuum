#pragma once

#include "ic/renderer/dx12_backend/dx12_device.h"

#include <d3d12.h>
#include <mutex>
#include <wrl/client.h>

namespace ic
{
    struct DX12DescriptorAllocation
    {
        D3D12_CPU_DESCRIPTOR_HANDLE cpuStart{};
        D3D12_GPU_DESCRIPTOR_HANDLE gpuStart{};

        uint32_t baseIndex = 0;
        uint32_t count = 0;
        uint32_t descriptorSize = 0;

        bool shaderVisible = false;

        bool valid() const
        {
            return count != 0;
        }
    };

    class DX12DescriptorHeap
    {
    public:
        void init(
            ID3D12Device* device,
            D3D12_DESCRIPTOR_HEAP_TYPE type,
            uint32_t capacity,
            bool shaderVisible);

        void shutdown();

        DX12DescriptorAllocation allocate(uint32_t count);

        ID3D12DescriptorHeap* heap() const
        {
            return m_heap.Get();
        }

        uint32_t descriptorSize() const
        {
            return m_descriptorSize;
        }

        uint32_t capacity() const
        {
            return m_capacity;
        }

        uint32_t allocated() const
        {
            return m_allocated;
        }

    private:
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_heap;

        D3D12_DESCRIPTOR_HEAP_TYPE m_type{};
        uint32_t m_capacity = 0;
        uint32_t m_allocated = 0;
        uint32_t m_descriptorSize = 0;
        bool m_shaderVisible = false;
        std::mutex m_mutex;
    };

    class DX12DescriptorSystem
    {
    public:
        struct Config
        {
            uint32_t rtvDescriptors = 4096;
            uint32_t dsvDescriptors = 1024;
            uint32_t resourceDescriptors = 1'000'000;
            uint32_t samplerDescriptors = 2048;
        };

        void init(
            const DX12Device& device,
            const Config& config = {});

        void shutdown();

        DX12DescriptorAllocation allocateRTV(uint32_t count);
        DX12DescriptorAllocation allocateDSV(uint32_t count);
        DX12DescriptorAllocation allocateResourceDescriptors(uint32_t count);
        DX12DescriptorAllocation allocateSamplers(uint32_t count);

        ID3D12DescriptorHeap* shaderResourceHeap() const;
        ID3D12DescriptorHeap* samplerHeap() const;

    private:
        Config m_config{};
        DX12DescriptorHeap m_rtvHeap;
        DX12DescriptorHeap m_dsvHeap;
        DX12DescriptorHeap m_cbvSrvUavHeap;
        DX12DescriptorHeap m_samplerHeap;
    };
}
