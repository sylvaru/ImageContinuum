#pragma once

#include "ic/renderer/render_types.h"
#include "ic/renderer/vulkan_backend/vulkan_common.h"

#include <mutex>
#include <utility>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

namespace ic
{
    struct VulkanBuffer
    {
        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VmaAllocationInfo allocationInfo{};
        VkDeviceSize size = 0;
        VkDeviceAddress gpuAddress = 0;
        void* mapped = nullptr;

        VulkanBuffer() = default;
        VulkanBuffer(const VulkanBuffer&) = delete;
        VulkanBuffer& operator=(const VulkanBuffer&) = delete;

        VulkanBuffer(VulkanBuffer&& other) noexcept
        {
            *this = std::move(other);
        }

        VulkanBuffer& operator=(VulkanBuffer&& other) noexcept
        {
            if (this != &other)
            {
                buffer = std::exchange(other.buffer, VK_NULL_HANDLE);
                allocation =
                    std::exchange(other.allocation, VK_NULL_HANDLE);
                allocationInfo =
                    std::exchange(other.allocationInfo, {});
                size = std::exchange(other.size, 0);
                gpuAddress = std::exchange(other.gpuAddress, 0);
                mapped = std::exchange(other.mapped, nullptr);
            }

            return *this;
        }

        explicit operator bool() const
        {
            return buffer != VK_NULL_HANDLE;
        }
    };

    struct VulkanTexture
    {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VmaAllocationInfo allocationInfo{};
        VkExtent3D extent{};
        VkFormat format = VK_FORMAT_UNDEFINED;
        uint32_t mipLevels = 1;
        uint32_t arrayLayers = 1;

        VulkanTexture() = default;
        VulkanTexture(const VulkanTexture&) = delete;
        VulkanTexture& operator=(const VulkanTexture&) = delete;

        VulkanTexture(VulkanTexture&& other) noexcept
        {
            *this = std::move(other);
        }

        VulkanTexture& operator=(VulkanTexture&& other) noexcept
        {
            if (this != &other)
            {
                image = std::exchange(other.image, VK_NULL_HANDLE);
                allocation =
                    std::exchange(other.allocation, VK_NULL_HANDLE);
                allocationInfo =
                    std::exchange(other.allocationInfo, {});
                extent = std::exchange(other.extent, {});
                format = std::exchange(other.format, VK_FORMAT_UNDEFINED);
                mipLevels = std::exchange(other.mipLevels, 1);
                arrayLayers = std::exchange(other.arrayLayers, 1);
            }

            return *this;
        }

        explicit operator bool() const
        {
            return image != VK_NULL_HANDLE;
        }
    };

	class VulkanResourceAllocator
	{
    public:
		void init(
			VkInstance instance,
			VkPhysicalDevice physicalDevice,
			VkDevice device,
            const VulkanDeviceInfo& info);

		void shutdown();

        VulkanBuffer createBuffer(const BufferDesc& desc);
        VulkanTexture createTexture(const TextureDesc& desc);

        void destroyBuffer(VulkanBuffer& buffer);
        void destroyTexture(VulkanTexture& texture);

        void* map(VulkanBuffer& buffer);
        void unmap(VulkanBuffer& buffer);
        void flush(VulkanBuffer& buffer, VkDeviceSize offset, VkDeviceSize size);

        VmaAllocator allocator() const
        {
            return m_allocator;
        }

    private:
        VkBufferUsageFlags toVkBufferUsage(BufferUsageFlags usage) const;
        VkImageUsageFlags toVkImageUsage(TextureUsageFlags usage) const;
        VkFormat toVkFormat(TextureFormat format) const;
        VmaMemoryUsage toVmaUsage(ResourceMemoryUsage usage) const;

        VkDevice m_device = VK_NULL_HANDLE;
        VmaAllocator m_allocator = VK_NULL_HANDLE;
        bool m_bufferDeviceAddress = false;
        std::mutex m_mutex;
	};
}
