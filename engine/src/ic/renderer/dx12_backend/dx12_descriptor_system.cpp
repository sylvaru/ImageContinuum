#include "ic/renderer/dx12_backend/dx12_descriptor_system.h"

#include <spdlog/spdlog.h>

#include <algorithm>
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
    void DX12DescriptorHeap::init(
        ID3D12Device* device,
        D3D12_DESCRIPTOR_HEAP_TYPE type,
        uint32_t capacity,
        bool shaderVisible)
    {
        if (!device)
        {
            throw std::runtime_error(
                "DX12DescriptorHeap requires a valid device.");
        }

        m_type = type;
        m_capacity = capacity;
        m_allocated = 0;
        m_shaderVisible = shaderVisible;

        D3D12_DESCRIPTOR_HEAP_DESC desc{};
        desc.Type = type;
        desc.NumDescriptors = capacity;
        desc.Flags = shaderVisible
            ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
            : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

        throwIfFailed(
            device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&m_heap)),
            "Failed to create DX12 descriptor heap.");

        m_descriptorSize =
            device->GetDescriptorHandleIncrementSize(type);
    }

    void DX12DescriptorHeap::shutdown()
    {
        std::scoped_lock lock(m_mutex);

        m_heap.Reset();
        m_type = {};
        m_capacity = 0;
        m_allocated = 0;
        m_descriptorSize = 0;
        m_shaderVisible = false;
        m_freeRanges.clear();
    }

    DX12DescriptorAllocation DX12DescriptorHeap::allocate(uint32_t count)
    {
        if (count == 0)
        {
            return {};
        }

        std::scoped_lock lock(m_mutex);

        if (!m_heap)
        {
            throw std::runtime_error(
                "DX12 descriptor heap exhausted.");
        }

        DX12DescriptorAllocation allocation{};
        allocation.count = count;
        allocation.descriptorSize = m_descriptorSize;
        allocation.shaderVisible = m_shaderVisible;

        for (size_t i = 0; i < m_freeRanges.size(); ++i)
        {
            FreeRange& range = m_freeRanges[i];
            if (range.count < count)
            {
                continue;
            }

            allocation.baseIndex = range.baseIndex;
            range.baseIndex += count;
            range.count -= count;

            if (range.count == 0)
            {
                m_freeRanges.erase(m_freeRanges.begin() + i);
            }

            allocation.cpuStart =
                m_heap->GetCPUDescriptorHandleForHeapStart();
            allocation.cpuStart.ptr +=
                static_cast<SIZE_T>(allocation.baseIndex) *
                m_descriptorSize;

            if (m_shaderVisible)
            {
                allocation.gpuStart =
                    m_heap->GetGPUDescriptorHandleForHeapStart();
                allocation.gpuStart.ptr +=
                    static_cast<UINT64>(allocation.baseIndex) *
                    m_descriptorSize;
            }

            return allocation;
        }

        if (m_allocated + count > m_capacity)
        {
            throw std::runtime_error(
                "DX12 descriptor heap exhausted.");
        }

        allocation.baseIndex = m_allocated;

        allocation.cpuStart =
            m_heap->GetCPUDescriptorHandleForHeapStart();
        allocation.cpuStart.ptr +=
            static_cast<SIZE_T>(allocation.baseIndex) *
            m_descriptorSize;

        if (m_shaderVisible)
        {
            allocation.gpuStart =
                m_heap->GetGPUDescriptorHandleForHeapStart();
            allocation.gpuStart.ptr +=
                static_cast<UINT64>(allocation.baseIndex) *
                m_descriptorSize;
        }

        m_allocated += count;
        return allocation;
    }

    void DX12DescriptorHeap::release(DX12DescriptorAllocation allocation)
    {
        if (!allocation.valid())
        {
            return;
        }

        std::scoped_lock lock(m_mutex);

        if (!m_heap ||
            allocation.baseIndex >= m_capacity ||
            allocation.baseIndex + allocation.count > m_capacity)
        {
            return;
        }

        m_freeRanges.push_back({
            .baseIndex = allocation.baseIndex,
            .count = allocation.count
        });

        std::sort(
            m_freeRanges.begin(),
            m_freeRanges.end(),
            [](const FreeRange& lhs, const FreeRange& rhs)
            {
                return lhs.baseIndex < rhs.baseIndex;
            });

        size_t write = 0;
        for (size_t read = 0; read < m_freeRanges.size(); ++read)
        {
            FreeRange range = m_freeRanges[read];
            if (range.count == 0)
            {
                continue;
            }

            if (write == 0)
            {
                m_freeRanges[write++] = range;
                continue;
            }

            FreeRange& previous = m_freeRanges[write - 1];
            const uint32_t previousEnd =
                previous.baseIndex + previous.count;

            if (range.baseIndex <= previousEnd)
            {
                const uint32_t rangeEnd =
                    range.baseIndex + range.count;
                previous.count =
                    std::max(previousEnd, rangeEnd) -
                    previous.baseIndex;
            }
            else
            {
                m_freeRanges[write++] = range;
            }
        }

        m_freeRanges.resize(write);
    }

    void DX12DescriptorSystem::init(
        const DX12Device& device,
        const Config& config)
    {
        if (!device.features().descriptorIndexing)
        {
            throw std::runtime_error(
                "DX12DescriptorSystem requires descriptor indexing support.");
        }

        m_config = config;

        m_rtvHeap.init(
            device.device(),
            D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
            m_config.rtvDescriptors,
            false);

        m_dsvHeap.init(
            device.device(),
            D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
            m_config.dsvDescriptors,
            false);

        m_cbvSrvUavHeap.init(
            device.device(),
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            m_config.resourceDescriptors,
            true);

        m_samplerHeap.init(
            device.device(),
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
            m_config.samplerDescriptors,
            true);

        spdlog::info(
            "[DX12DescriptorSystem] Initialized (rtv={}, dsv={}, resources={}, samplers={}, bindless={})",
            m_config.rtvDescriptors,
            m_config.dsvDescriptors,
            m_config.resourceDescriptors,
            m_config.samplerDescriptors,
            device.features().bindlessResources);
    }

    void DX12DescriptorSystem::shutdown()
    {
        m_samplerHeap.shutdown();
        m_cbvSrvUavHeap.shutdown();
        m_dsvHeap.shutdown();
        m_rtvHeap.shutdown();
        m_config = {};

        spdlog::info("[DX12DescriptorSystem] Shutdown");
    }

    DX12DescriptorAllocation DX12DescriptorSystem::allocateRTV(uint32_t count)
    {
        return m_rtvHeap.allocate(count);
    }

    DX12DescriptorAllocation DX12DescriptorSystem::allocateDSV(uint32_t count)
    {
        return m_dsvHeap.allocate(count);
    }

    DX12DescriptorAllocation DX12DescriptorSystem::allocateResourceDescriptors(uint32_t count)
    {
        return m_cbvSrvUavHeap.allocate(count);
    }

    DX12DescriptorAllocation DX12DescriptorSystem::allocateSamplers(uint32_t count)
    {
        return m_samplerHeap.allocate(count);
    }

    void DX12DescriptorSystem::releaseRTV(DX12DescriptorAllocation allocation)
    {
        m_rtvHeap.release(allocation);
    }

    void DX12DescriptorSystem::releaseDSV(DX12DescriptorAllocation allocation)
    {
        m_dsvHeap.release(allocation);
    }

    void DX12DescriptorSystem::releaseResourceDescriptors(
        DX12DescriptorAllocation allocation)
    {
        m_cbvSrvUavHeap.release(allocation);
    }

    void DX12DescriptorSystem::releaseSamplers(
        DX12DescriptorAllocation allocation)
    {
        m_samplerHeap.release(allocation);
    }

    ID3D12DescriptorHeap* DX12DescriptorSystem::shaderResourceHeap() const
    {
        return m_cbvSrvUavHeap.heap();
    }

    ID3D12DescriptorHeap* DX12DescriptorSystem::samplerHeap() const
    {
        return m_samplerHeap.heap();
    }
}
