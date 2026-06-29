#include "ic/renderer/vulkan_backend/vulkan_descriptor_system.h"

#include <spdlog/spdlog.h>

#include <stdexcept>

namespace ic
{
    void VulkanDescriptorSystem::init(
        VkDevice device,
        const VulkanDeviceInfo& info)
    {
        if (device == VK_NULL_HANDLE)
        {
            throw std::runtime_error(
                "VulkanDescriptorSystem requires a valid device.");
        }

        if (!info.supportedFeatures.descriptorIndexing ||
            !info.supportedFeatures.bufferDeviceAddress)
        {
            throw std::runtime_error(
                "VulkanDescriptorSystem requires descriptor indexing and buffer device address.");
        }

        m_device = device;

        if (info.supportedFeatures.descriptorHeap)
        {
            m_mode = VulkanDescriptorMode::DescriptorHeap;
            loadDescriptorHeapFunctions();
        }
        else if (info.supportedFeatures.descriptorBuffer)
        {
            m_mode = VulkanDescriptorMode::DescriptorBuffer;
            loadDescriptorBufferFunctions();
        }
        else
        {
            m_mode = VulkanDescriptorMode::DescriptorIndexing;
        }

        spdlog::info(
            "[VulkanDescriptorSystem] Initialized (mode={})",
            m_mode == VulkanDescriptorMode::DescriptorHeap
                ? "DescriptorHeap"
                : (m_mode == VulkanDescriptorMode::DescriptorBuffer
                    ? "DescriptorBuffer"
                    : "DescriptorIndexing"));
    }

    void VulkanDescriptorSystem::shutdown()
    {
        m_device = VK_NULL_HANDLE;
        m_mode = VulkanDescriptorMode::DescriptorIndexing;

        spdlog::info("[VulkanDescriptorSystem] Shutdown");
    }

    void VulkanDescriptorSystem::loadDescriptorHeapFunctions()
    {
#ifdef VK_EXT_descriptor_heap
        m_vkCmdBindResourceHeapEXT =
            reinterpret_cast<PFN_vkCmdBindResourceHeapEXT>(
                vkGetDeviceProcAddr(m_device, "vkCmdBindResourceHeapEXT"));
        m_vkCmdBindSamplerHeapEXT =
            reinterpret_cast<PFN_vkCmdBindSamplerHeapEXT>(
                vkGetDeviceProcAddr(m_device, "vkCmdBindSamplerHeapEXT"));
        m_vkWriteResourceDescriptorsEXT =
            reinterpret_cast<PFN_vkWriteResourceDescriptorsEXT>(
                vkGetDeviceProcAddr(m_device, "vkWriteResourceDescriptorsEXT"));
        m_vkWriteSamplerDescriptorsEXT =
            reinterpret_cast<PFN_vkWriteSamplerDescriptorsEXT>(
                vkGetDeviceProcAddr(m_device, "vkWriteSamplerDescriptorsEXT"));
        m_vkGetPhysicalDeviceDescriptorSizeEXT =
            reinterpret_cast<PFN_vkGetPhysicalDeviceDescriptorSizeEXT>(
                vkGetDeviceProcAddr(m_device, "vkGetPhysicalDeviceDescriptorSizeEXT"));

        if (!m_vkCmdBindResourceHeapEXT ||
            !m_vkCmdBindSamplerHeapEXT ||
            !m_vkWriteResourceDescriptorsEXT ||
            !m_vkWriteSamplerDescriptorsEXT ||
            !m_vkGetPhysicalDeviceDescriptorSizeEXT)
        {
            throw std::runtime_error(
                "VK_EXT_descriptor_heap was enabled but required function pointers are missing.");
        }
#else
        throw std::runtime_error(
            "VK_EXT_descriptor_heap was requested but headers do not expose it.");
#endif
    }

    void VulkanDescriptorSystem::loadDescriptorBufferFunctions()
    {
#ifdef VK_EXT_descriptor_buffer
        m_vkCmdBindDescriptorBuffersEXT =
            reinterpret_cast<PFN_vkCmdBindDescriptorBuffersEXT>(
                vkGetDeviceProcAddr(m_device, "vkCmdBindDescriptorBuffersEXT"));
        m_vkCmdSetDescriptorBufferOffsetsEXT =
            reinterpret_cast<PFN_vkCmdSetDescriptorBufferOffsetsEXT>(
                vkGetDeviceProcAddr(m_device, "vkCmdSetDescriptorBufferOffsetsEXT"));
        m_vkGetDescriptorEXT =
            reinterpret_cast<PFN_vkGetDescriptorEXT>(
                vkGetDeviceProcAddr(m_device, "vkGetDescriptorEXT"));

        if (!m_vkCmdBindDescriptorBuffersEXT ||
            !m_vkCmdSetDescriptorBufferOffsetsEXT ||
            !m_vkGetDescriptorEXT)
        {
            throw std::runtime_error(
                "VK_EXT_descriptor_buffer was enabled but required function pointers are missing.");
        }
#else
        throw std::runtime_error(
            "VK_EXT_descriptor_buffer was requested but headers do not expose it.");
#endif
    }
}
