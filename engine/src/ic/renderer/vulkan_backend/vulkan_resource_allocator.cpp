#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#define VMA_IMPLEMENTATION
#include "ic/renderer/vulkan_backend/vulkan_resource_allocator.h"
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include <spdlog/spdlog.h>

#include <stdexcept>

namespace
{
    void throwIfFailed(VkResult result, const char* message)
    {
        if (result != VK_SUCCESS)
        {
            throw std::runtime_error(message);
        }
    }
}

namespace ic
{
	void VulkanResourceAllocator::init(
        VkInstance instance,
        VkPhysicalDevice physicalDevice,
        VkDevice device,
        const VulkanDeviceInfo& info)
	{
        if (m_allocator != VK_NULL_HANDLE)
        {
            shutdown();
        }

        m_device = device;
        m_bufferDeviceAddress =
            info.supportedFeatures.bufferDeviceAddress;

        VmaAllocatorCreateInfo createInfo{};
        createInfo.instance = instance;
        createInfo.physicalDevice = physicalDevice;
        createInfo.device = device;
        createInfo.vulkanApiVersion = info.properties.apiVersion;

        if (m_bufferDeviceAddress)
        {
            createInfo.flags |=
                VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
        }

        throwIfFailed(
            vmaCreateAllocator(&createInfo, &m_allocator),
            "Failed to create Vulkan resource allocator.");

        spdlog::info(
            "[VulkanResourceAllocator] Initialized (bufferDeviceAddress={})",
            m_bufferDeviceAddress);
	}

	void VulkanResourceAllocator::shutdown()
	{
        std::scoped_lock lock(m_mutex);

        if (m_allocator != VK_NULL_HANDLE)
        {
            vmaDestroyAllocator(m_allocator);
            m_allocator = VK_NULL_HANDLE;
        }

        m_device = VK_NULL_HANDLE;
        m_bufferDeviceAddress = false;

        spdlog::info("[VulkanResourceAllocator] Shutdown");
	}

    VulkanBuffer VulkanResourceAllocator::createBuffer(const BufferDesc& desc)
    {
        if (desc.size == 0)
        {
            throw std::runtime_error("Cannot create zero-sized Vulkan buffer.");
        }

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = desc.size;
        bufferInfo.usage = toVkBufferUsage(desc.usage);
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (m_bufferDeviceAddress ||
            hasFlag(desc.usage, BufferUsageFlags::ShaderDeviceAddress))
        {
            bufferInfo.usage |=
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        }

        VmaAllocationCreateInfo allocationInfo{};
        allocationInfo.usage =
            toVmaUsage(desc.memoryUsage);

        if (desc.mappedAtCreation)
        {
            allocationInfo.flags |=
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
        }

        if (desc.memoryUsage == ResourceMemoryUsage::CpuToGpu)
        {
            allocationInfo.flags |=
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        }
        else if (desc.memoryUsage == ResourceMemoryUsage::GpuToCpu)
        {
            allocationInfo.flags |=
                VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
        }

        VulkanBuffer buffer{};
        buffer.size = desc.size;

        {
            std::scoped_lock lock(m_mutex);

            throwIfFailed(
                vmaCreateBuffer(
                    m_allocator,
                    &bufferInfo,
                    &allocationInfo,
                    &buffer.buffer,
                    &buffer.allocation,
                    &buffer.allocationInfo),
                "Failed to create Vulkan buffer.");
        }

        buffer.mapped = buffer.allocationInfo.pMappedData;

        if (m_bufferDeviceAddress &&
            hasFlag(desc.usage, BufferUsageFlags::ShaderDeviceAddress))
        {
            VkBufferDeviceAddressInfo addressInfo{};
            addressInfo.sType =
                VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
            addressInfo.buffer = buffer.buffer;

            buffer.gpuAddress =
                vkGetBufferDeviceAddress(m_device, &addressInfo);
        }

        if (desc.debugName)
        {
            vmaSetAllocationName(
                m_allocator,
                buffer.allocation,
                desc.debugName);
        }

        return buffer;
    }

    VulkanTexture VulkanResourceAllocator::createTexture(const TextureDesc& desc)
    {
        if (desc.width == 0 || desc.height == 0 || desc.depth == 0)
        {
            throw std::runtime_error("Cannot create zero-sized Vulkan texture.");
        }

        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType =
            desc.depth > 1 ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
        imageInfo.extent = {
            desc.width,
            desc.height,
            desc.depth
        };
        imageInfo.mipLevels = desc.mipLevels;
        imageInfo.arrayLayers = desc.arrayLayers;
        imageInfo.format = toVkFormat(desc.format);
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = toVkImageUsage(desc.usage);
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocationInfo{};
        allocationInfo.usage = toVmaUsage(desc.memoryUsage);

        VulkanTexture texture{};
        texture.extent = imageInfo.extent;
        texture.format = imageInfo.format;
        texture.mipLevels = desc.mipLevels;
        texture.arrayLayers = desc.arrayLayers;

        {
            std::scoped_lock lock(m_mutex);

            throwIfFailed(
                vmaCreateImage(
                    m_allocator,
                    &imageInfo,
                    &allocationInfo,
                    &texture.image,
                    &texture.allocation,
                    &texture.allocationInfo),
                "Failed to create Vulkan texture.");
        }

        if (desc.debugName)
        {
            vmaSetAllocationName(
                m_allocator,
                texture.allocation,
                desc.debugName);
        }

        return texture;
    }

    void VulkanResourceAllocator::destroyBuffer(VulkanBuffer& buffer)
    {
        if (!buffer)
        {
            return;
        }

        std::scoped_lock lock(m_mutex);

        vmaDestroyBuffer(
            m_allocator,
            buffer.buffer,
            buffer.allocation);

        buffer = {};
    }

    void VulkanResourceAllocator::destroyTexture(VulkanTexture& texture)
    {
        if (!texture)
        {
            return;
        }

        std::scoped_lock lock(m_mutex);

        vmaDestroyImage(
            m_allocator,
            texture.image,
            texture.allocation);

        texture = {};
    }

    void* VulkanResourceAllocator::map(VulkanBuffer& buffer)
    {
        if (!buffer)
        {
            return nullptr;
        }

        if (buffer.mapped)
        {
            return buffer.mapped;
        }

        std::scoped_lock lock(m_mutex);

        throwIfFailed(
            vmaMapMemory(
                m_allocator,
                buffer.allocation,
                &buffer.mapped),
            "Failed to map Vulkan buffer.");

        return buffer.mapped;
    }

    void VulkanResourceAllocator::unmap(VulkanBuffer& buffer)
    {
        if (!buffer || !buffer.mapped)
        {
            return;
        }

        std::scoped_lock lock(m_mutex);

        vmaUnmapMemory(
            m_allocator,
            buffer.allocation);

        buffer.mapped = nullptr;
    }

    void VulkanResourceAllocator::flush(
        VulkanBuffer& buffer,
        VkDeviceSize offset,
        VkDeviceSize size)
    {
        if (!buffer)
        {
            return;
        }

        std::scoped_lock lock(m_mutex);

        throwIfFailed(
            vmaFlushAllocation(
                m_allocator,
                buffer.allocation,
                offset,
                size),
            "Failed to flush Vulkan buffer allocation.");
    }

    VkBufferUsageFlags VulkanResourceAllocator::toVkBufferUsage(
        BufferUsageFlags usage) const
    {
        VkBufferUsageFlags flags = 0;

        if (hasFlag(usage, BufferUsageFlags::Vertex))
            flags |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        if (hasFlag(usage, BufferUsageFlags::Index))
            flags |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        if (hasFlag(usage, BufferUsageFlags::Constant))
            flags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        if (hasFlag(usage, BufferUsageFlags::Storage))
            flags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (hasFlag(usage, BufferUsageFlags::TransferSrc))
            flags |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (hasFlag(usage, BufferUsageFlags::TransferDst))
            flags |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (hasFlag(usage, BufferUsageFlags::Indirect))
            flags |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

        return flags == 0 ? VK_BUFFER_USAGE_TRANSFER_DST_BIT : flags;
    }

    VkImageUsageFlags VulkanResourceAllocator::toVkImageUsage(
        TextureUsageFlags usage) const
    {
        VkImageUsageFlags flags = 0;

        if (hasFlag(usage, TextureUsageFlags::Sampled))
            flags |= VK_IMAGE_USAGE_SAMPLED_BIT;
        if (hasFlag(usage, TextureUsageFlags::Storage))
            flags |= VK_IMAGE_USAGE_STORAGE_BIT;
        if (hasFlag(usage, TextureUsageFlags::ColorAttachment))
            flags |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (hasFlag(usage, TextureUsageFlags::DepthAttachment))
            flags |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        if (hasFlag(usage, TextureUsageFlags::TransferSrc))
            flags |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        if (hasFlag(usage, TextureUsageFlags::TransferDst))
            flags |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;

        return flags == 0 ? VK_IMAGE_USAGE_SAMPLED_BIT : flags;
    }

    VkFormat VulkanResourceAllocator::toVkFormat(TextureFormat format) const
    {
        switch (format)
        {
        case TextureFormat::RGBA8_UNorm:
            return VK_FORMAT_R8G8B8A8_UNORM;
        case TextureFormat::RGBA8_SRGB:
            return VK_FORMAT_R8G8B8A8_SRGB;
        case TextureFormat::RGBA32_Float:
            return VK_FORMAT_R32G32B32A32_SFLOAT;
        case TextureFormat::BGRA8_UNorm:
            return VK_FORMAT_B8G8R8A8_UNORM;
        case TextureFormat::BGRA8_SRGB:
            return VK_FORMAT_B8G8R8A8_SRGB;
        case TextureFormat::D32_Float:
            return VK_FORMAT_D32_SFLOAT;
        case TextureFormat::Unknown:
            break;
        }

        return VK_FORMAT_R8G8B8A8_UNORM;
    }

    VmaMemoryUsage VulkanResourceAllocator::toVmaUsage(
        ResourceMemoryUsage usage) const
    {
        switch (usage)
        {
        case ResourceMemoryUsage::GpuOnly:
            return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        case ResourceMemoryUsage::CpuToGpu:
            return VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        case ResourceMemoryUsage::GpuToCpu:
            return VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        }

        return VMA_MEMORY_USAGE_AUTO;
    }
}
