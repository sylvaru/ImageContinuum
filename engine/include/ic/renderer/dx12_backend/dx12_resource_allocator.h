#pragma once

#include "ic/renderer/dx12_backend/dx12_device.h"
#include "ic/renderer/render_types.h"

#include <d3d12.h>
#include <mutex>
#include <utility>
#include <wrl/client.h>

namespace ic
{
    struct ImageAsset;

    struct DX12Buffer
    {
        Microsoft::WRL::ComPtr<ID3D12Resource> resource;
        uint64_t size = 0;
        BufferUsageFlags usage = BufferUsageFlags::None;
        ResourceMemoryUsage memoryUsage = ResourceMemoryUsage::GpuOnly;
        bool mappedAtCreation = false;

        D3D12_RESOURCE_STATES initialState = D3D12_RESOURCE_STATE_COMMON;
        D3D12_GPU_VIRTUAL_ADDRESS gpuAddress = 0;
        void* mapped = nullptr;

        DX12Buffer() = default;
        DX12Buffer(const DX12Buffer&) = delete;
        DX12Buffer& operator=(const DX12Buffer&) = delete;

        DX12Buffer(DX12Buffer&& other) noexcept
        {
            moveFrom(other);
        }

        DX12Buffer& operator=(DX12Buffer&& other) noexcept
        {
            if (this != &other)
            {
                assert(resource == nullptr);
                moveFrom(other);
            }

            return *this;
        }

        explicit operator bool() const
        {
            return resource.Get() != nullptr;
        }

        void reset() noexcept
        {
            resource = nullptr;
            size = 0;
            initialState = D3D12_RESOURCE_STATE_COMMON;
            gpuAddress = 0;
            mapped = nullptr;
        }

    private:
        void moveFrom(DX12Buffer& other) noexcept
        {
            resource = std::exchange(other.resource, nullptr);
            size = std::exchange(other.size, 0);
            initialState = std::exchange(other.initialState, D3D12_RESOURCE_STATE_COMMON);
            gpuAddress = std::exchange(other.gpuAddress, 0);
            mapped = std::exchange(other.mapped, nullptr);
        }
    };

    struct DX12Texture
    {
        Microsoft::WRL::ComPtr<ID3D12Resource> resource;
        D3D12_RESOURCE_STATES initialState = D3D12_RESOURCE_STATE_COMMON;
        D3D12_RESOURCE_DESC desc{};

        DX12Texture() = default;
        DX12Texture(const DX12Texture&) = delete;
        DX12Texture& operator=(const DX12Texture&) = delete;

        DX12Texture(DX12Texture&& other) noexcept
        {
            moveFrom(other);
        }

        DX12Texture& operator=(DX12Texture&& other) noexcept
        {
            if (this != &other)
            {
                assert(resource == nullptr);
                moveFrom(other);
            }

            return *this;
        }

        explicit operator bool() const
        {
            return resource.Get() != nullptr;
        }

        void reset() noexcept
        {
            resource = nullptr;
            initialState = D3D12_RESOURCE_STATE_COMMON;
            desc = {};
        }
    private:
        void moveFrom(DX12Texture& other) noexcept
        {
            resource = std::exchange(other.resource, nullptr);
            initialState = std::exchange(other.initialState, D3D12_RESOURCE_STATE_COMMON);
            desc = std::exchange(other.desc, {});
        }
    };

    class DX12ResourceAllocator
    {
    public:
        void init(const DX12Device& device);
        void shutdown();

        DX12Buffer createBuffer(const BufferDesc& desc);
        DX12Texture createTexture(const TextureDesc& desc);
        DX12Texture createTexture(
            const ImageAsset& image,
            TextureUsageFlags usage =
                TextureUsageFlags::Sampled | TextureUsageFlags::TransferDst,
            const char* debugName = nullptr);

        void destroyBuffer(DX12Buffer& buffer);
        void destroyTexture(DX12Texture& texture);

        void* map(DX12Buffer& buffer);
        void unmap(DX12Buffer& buffer);

    private:
        D3D12_RESOURCE_FLAGS toBufferFlags(const BufferDesc& desc) const;
        D3D12_RESOURCE_FLAGS toTextureFlags(TextureUsageFlags usage) const;
        DXGI_FORMAT toDxgiFormat(TextureFormat format) const;
        D3D12_HEAP_PROPERTIES heapProperties(ResourceMemoryUsage usage) const;
        D3D12_RESOURCE_STATES initialBufferState(ResourceMemoryUsage usage) const;
        D3D12_RESOURCE_STATES initialTextureState(ResourceMemoryUsage usage) const;

        ID3D12Device5* m_device = nullptr;
        bool m_gpuVirtualAddress = false;
        std::mutex m_mutex;
    };
}
