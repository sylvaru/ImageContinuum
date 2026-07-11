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
        destroyCullBuffers();

        for (VulkanGpuSceneFrameResources& frame : m_frames)
        {
            m_resourceAllocator->destroyBuffer(frame.frameConstants);
            m_resourceAllocator->destroyBuffer(frame.objects);
            m_resourceAllocator->destroyBuffer(frame.materials);
            m_resourceAllocator->destroyBuffer(frame.visibleLights);
            m_resourceAllocator->destroyBuffer(frame.instanceBounds);
            m_resourceAllocator->destroyBuffer(frame.drawInputs);

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

    void VulkanGpuScene::ensureCullBuffers(
        uint32_t maxCullInstances,
        uint32_t maxBins)
    {
        destroyCullBuffers();

        maxInstances = maxCullInstances;

        visibleInstances = m_resourceAllocator->createBuffer({
            .size = maxCullInstances * sizeof(uint32_t),
            .usage = BufferUsageFlags::Storage,
            .memoryUsage = ResourceMemoryUsage::GpuOnly,
            .debugName = "Vulkan GPU cull visible instances"
        });
        visibleInstanceCount = m_resourceAllocator->createBuffer({
            .size = sizeof(uint32_t),
            .usage = BufferUsageFlags::Storage | BufferUsageFlags::TransferSrc,
            .memoryUsage = ResourceMemoryUsage::GpuOnly,
            .debugName = "Vulkan GPU cull visible count"
        });
        visibleInstanceCountReadback = m_resourceAllocator->createBuffer({
            .size = sizeof(uint32_t),
            .usage = BufferUsageFlags::TransferDst,
            .memoryUsage = ResourceMemoryUsage::GpuToCpu,
            .mappedAtCreation = true,
            .debugName = "Vulkan GPU cull visible count readback"
        });
        indirectArguments = m_resourceAllocator->createBuffer({
            .size = maxCullInstances * sizeof(GpuIndexedIndirectArguments),
            .usage = BufferUsageFlags::Storage | BufferUsageFlags::Indirect,
            .memoryUsage = ResourceMemoryUsage::GpuOnly,
            .debugName = "Vulkan GPU-driven indirect arguments"
        });
        drawMetadata = m_resourceAllocator->createBuffer({
            .size = maxCullInstances * sizeof(GpuDrawMetadata),
            .usage = BufferUsageFlags::Storage,
            .memoryUsage = ResourceMemoryUsage::GpuOnly,
            .debugName = "Vulkan GPU-driven draw metadata"
        });
        binCounts = m_resourceAllocator->createBuffer({
            .size = maxBins * sizeof(uint32_t),
            .usage = BufferUsageFlags::Storage | BufferUsageFlags::Indirect,
            .memoryUsage = ResourceMemoryUsage::GpuOnly,
            .debugName = "Vulkan GPU-driven bin counts"
        });
    }

    void VulkanGpuScene::destroyCullBuffers()
    {
        m_resourceAllocator->destroyBuffer(visibleInstances);
        m_resourceAllocator->destroyBuffer(visibleInstanceCount);
        m_resourceAllocator->destroyBuffer(visibleInstanceCountReadback);
        m_resourceAllocator->destroyBuffer(indirectArguments);
        m_resourceAllocator->destroyBuffer(drawMetadata);
        m_resourceAllocator->destroyBuffer(binCounts);
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
        const std::span<const GpuInstanceBounds> instanceBounds =
            m_prepared.instanceBounds();
        const std::span<const GpuDrawInput> drawInputs =
            m_prepared.drawInputs();

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

        if (visibleLights.size() > frameResources.visibleLightCapacity)
        {
            m_resourceAllocator->destroyBuffer(frameResources.visibleLights);
            frameResources.visibleLights =
                m_resourceAllocator->createBuffer({
                    .size = visibleLights.size() * sizeof(GpuVisibleLight),
                    .usage = BufferUsageFlags::Storage,
                    .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                    .mappedAtCreation = true,
                    .debugName = "Vulkan clustered visible lights"
                });
            frameResources.visibleLightCapacity =
                static_cast<uint32_t>(visibleLights.size());
            descriptorsDirty = true;
        }

        if (instanceBounds.size() > frameResources.instanceBoundsCapacity)
        {
            m_resourceAllocator->destroyBuffer(frameResources.instanceBounds);
            frameResources.instanceBounds =
                m_resourceAllocator->createBuffer({
                    .size = instanceBounds.size() * sizeof(GpuInstanceBounds),
                    .usage = BufferUsageFlags::Storage,
                    .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                    .mappedAtCreation = true,
                    .debugName = "Vulkan GPU cull instance bounds"
                });
            frameResources.instanceBoundsCapacity =
                static_cast<uint32_t>(instanceBounds.size());
        }
        if (drawInputs.size() > frameResources.drawInputCapacity)
        {
            m_resourceAllocator->destroyBuffer(frameResources.drawInputs);
            frameResources.drawInputs = m_resourceAllocator->createBuffer({
                .size = drawInputs.size() * sizeof(GpuDrawInput),
                .usage = BufferUsageFlags::Storage,
                .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                .mappedAtCreation = true,
                .debugName = "Vulkan GPU-driven draw inputs" });
            frameResources.drawInputCapacity =
                static_cast<uint32_t>(drawInputs.size());
        }

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
        std::memcpy(
            frameResources.visibleLights.mapped,
            visibleLights.data(),
            visibleLights.size() * sizeof(GpuVisibleLight));
        std::memcpy(
            frameResources.instanceBounds.mapped,
            instanceBounds.data(),
            instanceBounds.size() * sizeof(GpuInstanceBounds));
        std::memcpy(
            frameResources.drawInputs.mapped,
            drawInputs.data(),
            drawInputs.size() * sizeof(GpuDrawInput));

        m_resourceAllocator->flush(
            frameResources.frameConstants, 0, sizeof(frameData));
        m_resourceAllocator->flush(
            frameResources.objects, 0, objects.size() * sizeof(GpuObjectData));
        m_resourceAllocator->flush(
            frameResources.materials,
            0,
            materials.size() * sizeof(GpuMaterialData));
        m_resourceAllocator->flush(
            frameResources.visibleLights,
            0,
            visibleLights.size() * sizeof(GpuVisibleLight));
        m_resourceAllocator->flush(
            frameResources.instanceBounds,
            0,
            instanceBounds.size() * sizeof(GpuInstanceBounds));

        result.descriptorsDirty = descriptorsDirty;
        return result;
    }
}
