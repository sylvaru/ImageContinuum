#include "ic/renderer/vulkan_backend/vulkan_gpu_scene.h"

#include <algorithm>
#include <cstring>

namespace ic
{
    void VulkanGpuScene::init(
        VulkanResourceAllocator& resourceAllocator,
        uint32_t framesInFlight)
    {
        m_resourceAllocator = &resourceAllocator;
        m_frames.clear();
        m_frames.resize(std::max(1u, framesInFlight));
        m_prepared = PreparedGpuScene{};
    }

    void VulkanGpuScene::shutdown(VkDevice device)
    {
        for (VulkanGpuSceneFrameResources& frame : m_frames)
        {
            m_resourceAllocator->destroyBuffer(frame.frameConstants);
            m_resourceAllocator->destroyBuffer(frame.objects);
            m_resourceAllocator->destroyBuffer(frame.materials);

            if (device != VK_NULL_HANDLE)
            {
                if (frame.descriptorPool != VK_NULL_HANDLE)
                {
                    vkDestroyDescriptorPool(
                        device, frame.descriptorPool, nullptr);
                }
                if (frame.hiZDescriptorPool != VK_NULL_HANDLE)
                {
                    vkDestroyDescriptorPool(
                        device, frame.hiZDescriptorPool, nullptr);
                }
                if (frame.gpuCullDescriptorPool != VK_NULL_HANDLE)
                {
                    vkDestroyDescriptorPool(
                        device, frame.gpuCullDescriptorPool, nullptr);
                }
            }
        }
        m_frames.clear();

        m_resourceAllocator = nullptr;
    }

    VulkanGpuScene::PrepareResult VulkanGpuScene::prepare(
        uint64_t frameIndex,
        const SceneRenderView& scene,
        uint32_t frameSlot,
        const ResolveModelFn& resolveModel,
        const BuildFrameDataFn& buildFrameData)
    {
        PrepareResult result{};

        if (frameSlot >= m_frames.size())
        {
            return result;
        }

        const bool alreadyPreparedThisFrame =
            m_prepared.valid() && m_prepared.frameIndex() == frameIndex;

        if (!m_prepared.build(frameIndex, scene, resolveModel))
        {
            return result;
        }

        result.hasData = true;

        if (alreadyPreparedThisFrame)
        {
            return result;
        }

        const std::span<const GpuObjectData> objects = m_prepared.objects();
        const std::span<const GpuMaterialData> materials =
            m_prepared.materials();
        const std::span<const GpuVisibleLight> visibleLights =
            m_prepared.visibleLights();
        // Only the count is needed here (for GpuFrameData); the bounds/inputs
        // themselves are uploaded by the backend into the graph-owned buffers.
        const std::span<const GpuInstanceBounds> instanceBounds =
            m_prepared.instanceBounds();

        VulkanGpuSceneFrameResources& frameResources = m_frames[frameSlot];

        if (!frameResources.frameConstants)
        {
            frameResources.frameConstants =
                m_resourceAllocator->createBuffer({
                    .size = sizeof(GpuFrameData),
                    .usage = BufferUsageFlags::Constant,
                    .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                    .mappedAtCreation = true,
                    .debugName = "Vulkan frame constants"
                });
        }

        bool descriptorsDirty = false;
        if (objects.size() > frameResources.objectCapacity)
        {
            m_resourceAllocator->destroyBuffer(frameResources.objects);
            frameResources.objects =
                m_resourceAllocator->createBuffer({
                    .size = objects.size() * sizeof(GpuObjectData),
                    .usage = BufferUsageFlags::Storage,
                    .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                    .mappedAtCreation = true,
                    .debugName = "Vulkan object data"
                });
            frameResources.objectCapacity =
                static_cast<uint32_t>(objects.size());
            descriptorsDirty = true;
        }

        if (materials.size() > frameResources.materialCapacity)
        {
            m_resourceAllocator->destroyBuffer(frameResources.materials);
            frameResources.materials =
                m_resourceAllocator->createBuffer({
                    .size = materials.size() * sizeof(GpuMaterialData),
                    .usage = BufferUsageFlags::Storage,
                    .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                    .mappedAtCreation = true,
                    .debugName = "Vulkan material data"
                });
            frameResources.materialCapacity =
                static_cast<uint32_t>(materials.size());
            descriptorsDirty = true;
        }

        // visibleLights / instanceBounds / drawInputs are uploaded by the
        // backend into the graph-owned per-frame-slot buffers, not here.
        const GpuFrameData frameData = buildFrameData(
            static_cast<uint32_t>(visibleLights.size()),
            static_cast<uint32_t>(instanceBounds.size()),
            static_cast<uint32_t>(m_prepared.geometryBins().size()));

        std::memcpy(
            frameResources.frameConstants.mapped,
            &frameData,
            sizeof(frameData));
        std::memcpy(
            frameResources.objects.mapped,
            objects.data(),
            objects.size() * sizeof(GpuObjectData));
        std::memcpy(
            frameResources.materials.mapped,
            materials.data(),
            materials.size() * sizeof(GpuMaterialData));

        m_resourceAllocator->flush(
            frameResources.frameConstants, 0, sizeof(frameData));
        m_resourceAllocator->flush(
            frameResources.objects, 0, objects.size() * sizeof(GpuObjectData));
        m_resourceAllocator->flush(
            frameResources.materials,
            0,
            materials.size() * sizeof(GpuMaterialData));

        result.descriptorsDirty = descriptorsDirty;
        return result;
    }
}
