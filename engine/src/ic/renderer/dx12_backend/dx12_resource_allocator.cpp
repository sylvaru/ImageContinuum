#include "ic/renderer/dx12_backend/dx12_resource_allocator.h"

#include "ic/core/asset_manager.h"

#include <spdlog/spdlog.h>

#include <string>
#include <stdexcept>

namespace
{
    void throwIfFailed(HRESULT hr, const char* message)
    {
        if (FAILED(hr))
        {
            throw std::runtime_error(
                std::string(message) + " (HRESULT=" +
                std::to_string(static_cast<uint32_t>(hr)) + ")");
        }
    }

    ic::TextureFormat textureFormatFromImageAsset(const ic::ImageAsset& image)
    {
        switch (image.format)
        {
        case ic::ImageFormat::RGBA8:
            return image.srgb
                ? ic::TextureFormat::RGBA8_SRGB
                : ic::TextureFormat::RGBA8_UNorm;
        case ic::ImageFormat::RGBA32F:
            return ic::TextureFormat::RGBA32_Float;
        case ic::ImageFormat::R8:
        case ic::ImageFormat::RG8:
        case ic::ImageFormat::RGB8:
        case ic::ImageFormat::Unknown:
            break;
        }

        throw std::runtime_error(
            "ImageAsset format cannot be represented by the DX12 texture allocator. "
            "Decode with forceRGBA=true or add a renderer TextureFormat mapping.");
    }
}

namespace ic
{
    void DX12ResourceAllocator::init(const DX12Device& device)
    {
        m_device = device.device();
        m_gpuVirtualAddress = device.features().gpuVirtualAddress;

        if (!m_device)
        {
            throw std::runtime_error(
                "DX12ResourceAllocator requires a valid device.");
        }

        spdlog::info(
            "[DX12ResourceAllocator] Initialized (gpuVA={})",
            m_gpuVirtualAddress);
    }

    void DX12ResourceAllocator::shutdown()
    {
        std::scoped_lock lock(m_mutex);
        m_device = nullptr;
        m_gpuVirtualAddress = false;

        spdlog::info("[DX12ResourceAllocator] Shutdown");
    }

    DX12Buffer DX12ResourceAllocator::createBuffer(const BufferDesc& desc)
    {
        if (desc.size == 0)
        {
            throw std::runtime_error("Cannot create zero-sized DX12 buffer.");
        }

        D3D12_RESOURCE_DESC resourceDesc{};
        resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        resourceDesc.Alignment = 0;
        resourceDesc.Width = desc.size;
        resourceDesc.Height = 1;
        resourceDesc.DepthOrArraySize = 1;
        resourceDesc.MipLevels = 1;
        resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
        resourceDesc.SampleDesc.Count = 1;
        resourceDesc.SampleDesc.Quality = 0;
        resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        resourceDesc.Flags = toBufferFlags(desc);

        D3D12_HEAP_PROPERTIES heap =
            heapProperties(desc.memoryUsage);

        const D3D12_RESOURCE_STATES initialState =
            initialBufferState(desc.memoryUsage);

        DX12Buffer buffer{};
        buffer.size = desc.size;
        buffer.usage = desc.usage;
        buffer.memoryUsage = desc.memoryUsage;
        buffer.mappedAtCreation = desc.mappedAtCreation;
        buffer.initialState = initialState;
       
        {
            std::scoped_lock lock(m_mutex);

            throwIfFailed(
                m_device->CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &resourceDesc,
                    initialState,
                    nullptr,
                    IID_PPV_ARGS(&buffer.resource)),
                "Failed to create DX12 buffer.");
        }

        if (desc.debugName)
        {
            const std::string name(desc.debugName);
            const std::wstring wideName(name.begin(), name.end());
            buffer.resource->SetName(wideName.c_str());
        }

        if (m_gpuVirtualAddress &&
            desc.memoryUsage != ResourceMemoryUsage::GpuToCpu)
        {
            buffer.gpuAddress =
                buffer.resource->GetGPUVirtualAddress();
        }

        if (desc.mappedAtCreation)
        {
            buffer.mapped = map(buffer);
        }

        return buffer;
    }

    DX12Texture DX12ResourceAllocator::createTexture(const TextureDesc& desc)
    {
        if (desc.width == 0 || desc.height == 0 || desc.depth == 0)
        {
            throw std::runtime_error("Cannot create zero-sized DX12 texture.");
        }

        D3D12_RESOURCE_DESC resourceDesc{};
        resourceDesc.Dimension =
            desc.depth > 1
                ? D3D12_RESOURCE_DIMENSION_TEXTURE3D
                : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        resourceDesc.Alignment = 0;
        resourceDesc.Width = desc.width;
        resourceDesc.Height = desc.height;
        resourceDesc.DepthOrArraySize =
            static_cast<UINT16>(
                desc.depth > 1 ? desc.depth : desc.arrayLayers);
        resourceDesc.MipLevels =
            static_cast<UINT16>(desc.mipLevels);
        resourceDesc.Format =
            desc.format == TextureFormat::D32_Float &&
                hasFlag(desc.usage, TextureUsageFlags::Sampled)
                ? DXGI_FORMAT_R32_TYPELESS
                : toDxgiFormat(desc.format);
        resourceDesc.SampleDesc.Count = 1;
        resourceDesc.SampleDesc.Quality = 0;
        resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        resourceDesc.Flags = toTextureFlags(desc.usage);

        D3D12_HEAP_PROPERTIES heap =
            heapProperties(desc.memoryUsage);

        const D3D12_RESOURCE_STATES initialState =
            initialTextureState(desc.memoryUsage);

        D3D12_CLEAR_VALUE clearValue{};
        D3D12_CLEAR_VALUE* optimizedClearValue = nullptr;

        if (hasFlag(desc.usage, TextureUsageFlags::DepthAttachment))
        {
            clearValue.Format = toDxgiFormat(desc.format);
            clearValue.DepthStencil.Depth = 1.0f;
            clearValue.DepthStencil.Stencil = 0;
            optimizedClearValue = &clearValue;
        }
        else if (hasFlag(desc.usage, TextureUsageFlags::ColorAttachment))
        {
            clearValue.Format = resourceDesc.Format;
            clearValue.Color[0] = 0.02f;
            clearValue.Color[1] = 0.02f;
            clearValue.Color[2] = 0.025f;
            clearValue.Color[3] = 1.0f;
            optimizedClearValue = &clearValue;
        }

        DX12Texture texture{};
        texture.initialState = initialState;
        texture.desc = resourceDesc;

        {
            std::scoped_lock lock(m_mutex);

            throwIfFailed(
                m_device->CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &resourceDesc,
                    initialState,
                    optimizedClearValue,
                    IID_PPV_ARGS(&texture.resource)),
                "Failed to create DX12 texture.");
        }

        if (desc.debugName)
        {
            const std::string name(desc.debugName);
            const std::wstring wideName(name.begin(), name.end());
            texture.resource->SetName(wideName.c_str());
        }

        return texture;
    }

    DX12Texture DX12ResourceAllocator::createTexture(
        const ImageAsset& image,
        TextureUsageFlags usage,
        const char* debugName)
    {
        if (!image.valid())
        {
            throw std::runtime_error("Cannot create DX12 texture from invalid ImageAsset.");
        }

        TextureDesc desc{};
        desc.width = image.width;
        desc.height = image.height;
        desc.depth = 1;
        desc.mipLevels = 1;
        desc.arrayLayers = 1;
        desc.format = textureFormatFromImageAsset(image);
        desc.usage = usage;
        desc.memoryUsage = ResourceMemoryUsage::GpuOnly;
        desc.debugName = debugName;
        return createTexture(desc);
    }

    void DX12ResourceAllocator::destroyBuffer(DX12Buffer& buffer)
    {
        if (!buffer) return;

        if (buffer.mapped) unmap(buffer);

        buffer.reset();
    }

    void DX12ResourceAllocator::destroyTexture(DX12Texture& texture)
    {
        if (!texture) return;

        texture.reset();
    }

    void* DX12ResourceAllocator::map(DX12Buffer& buffer)
    {
        if (!buffer)
        {
            return nullptr;
        }

        if (buffer.memoryUsage == ResourceMemoryUsage::GpuOnly)
        {
            throw std::runtime_error(
                "Cannot map GPU only buffer");
        }

        if (buffer.mapped)
        {
            return buffer.mapped;
        }

        D3D12_RANGE readRange{};
        throwIfFailed(
            buffer.resource->Map(
                0,
                &readRange,
                &buffer.mapped),
            "Failed to map DX12 buffer.");

        return buffer.mapped;
    }

    void DX12ResourceAllocator::unmap(DX12Buffer& buffer)
    {
        if (!buffer || !buffer.mapped)
        {
            return;
        }

        buffer.resource->Unmap(0, nullptr);
        buffer.mapped = nullptr;
    }

    D3D12_RESOURCE_FLAGS DX12ResourceAllocator::toBufferFlags(
        const BufferDesc& desc) const
    {
        D3D12_RESOURCE_FLAGS flags =
            D3D12_RESOURCE_FLAG_NONE;

        // Only GPU-only storage buffers become UAVs. A non-GPU-only storage
        // buffer lives on the upload heap (which cannot carry the UAV flag) and
        // is a shader-readable structured buffer bound as an SRV, such as the
        // graph-owned, CPU-uploaded GPU-driven cull inputs. Vulkan expresses the
        // same buffers with VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, so the shared
        // API-neutral declaration carries BufferUsageFlags::Storage for both.
        if (hasFlag(desc.usage, BufferUsageFlags::Storage) &&
            desc.memoryUsage == ResourceMemoryUsage::GpuOnly)
        {
            flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        }

        return flags;
    }

    D3D12_RESOURCE_FLAGS DX12ResourceAllocator::toTextureFlags(
        TextureUsageFlags usage) const
    {
        D3D12_RESOURCE_FLAGS flags =
            D3D12_RESOURCE_FLAG_NONE;

        if (hasFlag(usage, TextureUsageFlags::ColorAttachment))
        {
            flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        }

        if (hasFlag(usage, TextureUsageFlags::DepthAttachment))
        {
            flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        }

        if (hasFlag(usage, TextureUsageFlags::Storage))
        {
            flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        }

        return flags;
    }

    DXGI_FORMAT DX12ResourceAllocator::toDxgiFormat(TextureFormat format) const
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

        return DXGI_FORMAT_R8G8B8A8_UNORM;
    }

    D3D12_HEAP_PROPERTIES DX12ResourceAllocator::heapProperties(
        ResourceMemoryUsage usage) const
    {
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        heap.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heap.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heap.CreationNodeMask = 1;
        heap.VisibleNodeMask = 1;

        switch (usage)
        {
        case ResourceMemoryUsage::GpuOnly:
            heap.Type = D3D12_HEAP_TYPE_DEFAULT;
            break;
        case ResourceMemoryUsage::CpuToGpu:
            heap.Type = D3D12_HEAP_TYPE_UPLOAD;
            break;
        case ResourceMemoryUsage::GpuToCpu:
            heap.Type = D3D12_HEAP_TYPE_READBACK;
            break;
        }

        return heap;
    }

    D3D12_RESOURCE_STATES DX12ResourceAllocator::initialBufferState(
        ResourceMemoryUsage usage) const
    {
        switch (usage)
        {
        case ResourceMemoryUsage::CpuToGpu:
            return D3D12_RESOURCE_STATE_GENERIC_READ;
        case ResourceMemoryUsage::GpuToCpu:
            return D3D12_RESOURCE_STATE_COPY_DEST;
        case ResourceMemoryUsage::GpuOnly:
            return D3D12_RESOURCE_STATE_COMMON;
        }

        return D3D12_RESOURCE_STATE_COMMON;
    }

    D3D12_RESOURCE_STATES DX12ResourceAllocator::initialTextureState(
        ResourceMemoryUsage usage) const
    {
        if (usage != ResourceMemoryUsage::GpuOnly)
        {
            throw std::runtime_error(
                "DX12 textures must currently be created in GPU-only memory.");
        }

        return D3D12_RESOURCE_STATE_COMMON;
    }
}
