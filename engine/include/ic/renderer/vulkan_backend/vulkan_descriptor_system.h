#pragma once

#include "ic/renderer/vulkan_backend/vulkan_common.h"

#include <vulkan/vulkan.h>

namespace ic
{
    enum class VulkanDescriptorMode : uint8_t
    {
        DescriptorIndexing,
        DescriptorBuffer,
        DescriptorHeap
    };

    class VulkanDescriptorSystem
    {
    public:
        void init(
            VkDevice device,
            const VulkanDeviceInfo& info);

        void shutdown();

        VulkanDescriptorMode mode() const
        {
            return m_mode;
        }

        bool usesDescriptorHeap() const
        {
            return m_mode == VulkanDescriptorMode::DescriptorHeap;
        }

        bool usesDescriptorBuffer() const
        {
            return m_mode == VulkanDescriptorMode::DescriptorBuffer;
        }

    private:
        void loadDescriptorHeapFunctions();
        void loadDescriptorBufferFunctions();

        VkDevice m_device = VK_NULL_HANDLE;
        VulkanDescriptorMode m_mode = VulkanDescriptorMode::DescriptorIndexing;

#ifdef VK_EXT_descriptor_heap
        PFN_vkCmdBindResourceHeapEXT m_vkCmdBindResourceHeapEXT = nullptr;
        PFN_vkCmdBindSamplerHeapEXT m_vkCmdBindSamplerHeapEXT = nullptr;
        PFN_vkWriteResourceDescriptorsEXT m_vkWriteResourceDescriptorsEXT = nullptr;
        PFN_vkWriteSamplerDescriptorsEXT m_vkWriteSamplerDescriptorsEXT = nullptr;
        PFN_vkGetPhysicalDeviceDescriptorSizeEXT m_vkGetPhysicalDeviceDescriptorSizeEXT = nullptr;
#endif

#ifdef VK_EXT_descriptor_buffer
        PFN_vkCmdBindDescriptorBuffersEXT m_vkCmdBindDescriptorBuffersEXT = nullptr;
        PFN_vkCmdSetDescriptorBufferOffsetsEXT m_vkCmdSetDescriptorBufferOffsetsEXT = nullptr;
        PFN_vkGetDescriptorEXT m_vkGetDescriptorEXT = nullptr;
#endif
    };
}
