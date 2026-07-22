#include "ic/common/ic_pch.h"
#include "ic/renderer/vulkan_backend/vulkan_acceleration_structure_provider.h"

#include <algorithm>
#include <cstring>
#include <unordered_set>
#include <spdlog/spdlog.h>

namespace ic
{
    namespace
    {
        VkGeometryFlagsKHR nativeGeometryFlags(uint32_t flags)
        {
            return (flags & RayTracingGeometryOpaque) != 0
                ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0u;
        }

        VkTransformMatrixKHR nativeTransform(const glm::mat4& value)
        {
            VkTransformMatrixKHR result{};
            for (uint32_t row = 0; row < 3; ++row)
                for (uint32_t column = 0; column < 4; ++column)
                    result.matrix[row][column] = value[column][row];
            return result;
        }
    }

    void VulkanAccelerationStructureProvider::init(
        const VulkanDevice& device, VulkanResourceAllocator& allocator,
        VulkanRetirementQueue& retirement, uint32_t framesInFlight)
    {
        m_device = device.device();
        m_createAs = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(
            vkGetDeviceProcAddr(m_device, "vkCreateAccelerationStructureKHR"));
        m_destroyAs = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(
            vkGetDeviceProcAddr(m_device, "vkDestroyAccelerationStructureKHR"));
        m_getBuildSizes = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(
            vkGetDeviceProcAddr(m_device, "vkGetAccelerationStructureBuildSizesKHR"));
        m_cmdBuild = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(
            vkGetDeviceProcAddr(m_device, "vkCmdBuildAccelerationStructuresKHR"));
        m_getAddress = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
            vkGetDeviceProcAddr(m_device, "vkGetAccelerationStructureDeviceAddressKHR"));
        m_allocator = &allocator;
        m_retirement = &retirement;
        m_hardware = device.info().supportedFeatures.accelerationStructure &&
            device.info().supportedFeatures.rayQuery && m_createAs &&
            m_destroyAs && m_getBuildSizes && m_cmdBuild && m_getAddress;
        m_instanceBuffers.resize(std::max(1u, framesInFlight));
        m_statistics.state = m_hardware
            ? RayTracingAccelerationStructureState::Empty
            : RayTracingAccelerationStructureState::Unsupported;
    }

    void VulkanAccelerationStructureProvider::shutdown()
    {
        if (!m_allocator) return;
        for (Blas& blas : m_blas)
        {
            if (blas.as && m_destroyAs) m_destroyAs(m_device, blas.as, nullptr);
            m_allocator->destroyBuffer(blas.storage);
        }
        m_blas.clear();
        if (m_tlas && m_destroyAs) m_destroyAs(m_device, m_tlas, nullptr);
        m_tlas = VK_NULL_HANDLE;
        m_allocator->destroyBuffer(m_tlasStorage);
        m_allocator->destroyBuffer(m_scratch);
        for (VulkanBuffer& buffer : m_instanceBuffers)
            m_allocator->destroyBuffer(buffer);
        m_instanceBuffers.clear();
        m_allocator = nullptr;
        m_retirement = nullptr;
        m_device = VK_NULL_HANDLE;
    }

    void VulkanAccelerationStructureProvider::retire(VulkanBuffer& buffer)
    {
        if (!buffer) return;
        m_retirement->retire(std::move(buffer));
        ++m_statistics.retiredResourceCount;
    }

    void VulkanAccelerationStructureProvider::retire(
        VkAccelerationStructureKHR& accelerationStructure)
    {
        if (!accelerationStructure) return;
        m_retirement->retireAccelerationStructure(accelerationStructure);
        accelerationStructure = VK_NULL_HANDLE;
        ++m_statistics.retiredResourceCount;
    }

    void VulkanAccelerationStructureProvider::ensureBuffer(
        VulkanBuffer& buffer, VkDeviceSize size, BufferUsageFlags usage,
        ResourceMemoryUsage memory, bool mapped, const char* name)
    {
        size = std::max<VkDeviceSize>(size, 256u);
        if (buffer && buffer.size >= size) return;
        retire(buffer);
        buffer = m_allocator->createBuffer({ .size = size, .usage = usage,
            .memoryUsage = memory, .mappedAtCreation = mapped,
            .debugName = name });
    }

    VkAccelerationStructureKHR VulkanAccelerationStructureProvider::createAs(
        VkAccelerationStructureTypeKHR type, const VulkanBuffer& storage,
        VkDeviceSize size)
    {
        VkAccelerationStructureCreateInfoKHR info{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
        info.buffer = storage.buffer;
        info.size = size;
        info.type = type;
        VkAccelerationStructureKHR result = VK_NULL_HANDLE;
        if (!m_createAs || m_createAs(m_device, &info, nullptr, &result)
            != VK_SUCCESS)
            return VK_NULL_HANDLE;
        return result;
    }

    void VulkanAccelerationStructureProvider::setEnabled(bool enabled)
    {
        if (m_enabled == enabled) return;
        m_enabled = enabled;
        invalidate();
        m_statistics.state = !enabled
            ? RayTracingAccelerationStructureState::Disabled
            : (m_hardware ? RayTracingAccelerationStructureState::Empty
                          : RayTracingAccelerationStructureState::Unsupported);
    }

    void VulkanAccelerationStructureProvider::invalidate() noexcept
    {
        m_prepared = false;
        m_recorded = false;
        m_hasTlas = false;
        m_blasValid = false;
        m_statistics.sceneGeneration = 0;
        if (m_enabled)
            m_statistics.state = m_hardware
                ? RayTracingAccelerationStructureState::Empty
                : RayTracingAccelerationStructureState::Unsupported;
    }

    RayTracingCapabilities VulkanAccelerationStructureProvider::capabilities()
        const noexcept
    {
        return { .accelerationStructures = m_hardware,
            .inlineRayQueries = m_hardware,
            .accelerationStructureUpdates = m_hardware,
            .indirectDispatch = true,
            .sharedAccelerationStructures = m_enabled && m_recorded &&
                m_statistics.state == RayTracingAccelerationStructureState::Ready };
    }

    bool VulkanAccelerationStructureProvider::readyFor(uint64_t generation)
        const noexcept
    {
        return capabilities().sharedAccelerationStructures &&
            m_statistics.sceneGeneration == generation;
    }

    uint64_t VulkanAccelerationStructureProvider::shaderTlasHandle() const noexcept
    {
        if (!readyFor(m_statistics.sceneGeneration)) return 0;
        VkAccelerationStructureDeviceAddressInfoKHR info{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR };
        info.accelerationStructure = m_tlas;
        return m_getAddress ? m_getAddress(m_device, &info) : 0;
    }

    bool VulkanAccelerationStructureProvider::prepare(
        const RayTracingSceneService& scene,
        const std::unordered_map<AssetHandle, VulkanUploadedModel,
            AssetHandleHash>& models)
    {
        if ((m_prepared || m_recorded) &&
            m_statistics.sceneGeneration == scene.statistics().generation)
            return m_prepared;
        if (!m_enabled || !m_hardware || !scene.statistics().valid ||
            scene.geometries().empty() || scene.instances().empty())
        {
            m_recorded = false;
            m_prepared = false;
            m_statistics.state = !m_enabled
                ? RayTracingAccelerationStructureState::Disabled
                : (m_hardware ? RayTracingAccelerationStructureState::Empty
                              : RayTracingAccelerationStructureState::Unsupported);
            return false;
        }

        const auto geometries = scene.geometries();
        const bool geometryRebuild = !m_blasValid ||
            scene.lastUpdateKind() == RayTracingSceneUpdateKind::GeometryRebuild ||
            m_blas.size() != geometries.size();
        if (geometryRebuild)
        {
            for (Blas& blas : m_blas) { retire(blas.as); retire(blas.storage); }
            m_blas.clear();
            m_blas.resize(geometries.size());
            m_blasValid = false;
        }

        VkDeviceSize maxScratch = 0;
        uint64_t resultBytes = 0;
        for (size_t i = 0; i < geometries.size(); ++i)
        {
            const auto modelIt = models.find(geometries[i].model);
            if (modelIt == models.end() || !modelIt->second.uploaded) return false;
            const VulkanUploadedModel& model = modelIt->second;
            VkAccelerationStructureGeometryTrianglesDataKHR triangles{
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR };
            triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            triangles.vertexData.deviceAddress = model.vertexBuffer.gpuAddress;
            triangles.vertexStride = sizeof(AssetVertex);
            triangles.maxVertex = static_cast<uint32_t>(model.vertexBuffer.size /
                sizeof(AssetVertex));
            triangles.indexType = VK_INDEX_TYPE_UINT32;
            triangles.indexData.deviceAddress = model.indexBuffer.gpuAddress +
                static_cast<VkDeviceSize>(geometries[i].firstIndex) * sizeof(uint32_t);
            VkAccelerationStructureGeometryKHR geometry{
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
            geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            geometry.flags = nativeGeometryFlags(geometries[i].flags);
            geometry.geometry.triangles = triangles;
            const uint32_t primitiveCount = geometries[i].indexCount / 3u;
            VkAccelerationStructureBuildGeometryInfoKHR build{
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
            build.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
            build.geometryCount = 1;
            build.pGeometries = &geometry;
            VkAccelerationStructureBuildSizesInfoKHR sizes{
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
            m_getBuildSizes(m_device,
                VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build,
                &primitiveCount, &sizes);
            ensureBuffer(m_blas[i].storage, sizes.accelerationStructureSize,
                BufferUsageFlags::AccelerationStructureStorage |
                    BufferUsageFlags::ShaderDeviceAddress,
                ResourceMemoryUsage::GpuOnly, false, "Vulkan shared RT BLAS");
            if (!m_blas[i].as)
                m_blas[i].as = createAs(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                    m_blas[i].storage, sizes.accelerationStructureSize);
            if (!m_blas[i].as) return false;
            maxScratch = std::max(maxScratch, sizes.buildScratchSize);
            resultBytes += sizes.accelerationStructureSize;
        }

        VkAccelerationStructureGeometryInstancesDataKHR instances{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR };
        VkAccelerationStructureGeometryKHR tlasGeometry{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
        tlasGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        tlasGeometry.geometry.instances = instances;
        VkAccelerationStructureBuildGeometryInfoKHR build{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
        build.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
            VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
        build.geometryCount = 1; build.pGeometries = &tlasGeometry;
        const uint32_t count = static_cast<uint32_t>(scene.instances().size());
        VkAccelerationStructureBuildSizesInfoKHR sizes{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
        m_getBuildSizes(m_device,
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build, &count, &sizes);
        const bool tlasGrowth = !m_tlasStorage ||
            m_tlasStorage.size < sizes.accelerationStructureSize;
        if (tlasGrowth) { retire(m_tlas); retire(m_tlasStorage); }
        ensureBuffer(m_tlasStorage, sizes.accelerationStructureSize,
            BufferUsageFlags::AccelerationStructureStorage |
                BufferUsageFlags::ShaderDeviceAddress,
            ResourceMemoryUsage::GpuOnly, false, "Vulkan shared RT TLAS");
        if (!m_tlas)
            m_tlas = createAs(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                m_tlasStorage, sizes.accelerationStructureSize);
        ensureBuffer(m_scratch, std::max(maxScratch,
                std::max(sizes.buildScratchSize, sizes.updateScratchSize)),
            BufferUsageFlags::Storage |
                BufferUsageFlags::AccelerationStructureScratch |
                BufferUsageFlags::ShaderDeviceAddress,
            ResourceMemoryUsage::GpuOnly, false, "Vulkan shared RT scratch");
        if (count > m_instanceCapacity)
        {
            m_instanceCapacity = std::max(count, std::max(64u, m_instanceCapacity * 2u));
            for (VulkanBuffer& buffer : m_instanceBuffers)
                ensureBuffer(buffer, VkDeviceSize(m_instanceCapacity) *
                    sizeof(VkAccelerationStructureInstanceKHR),
                    BufferUsageFlags::AccelerationStructureBuildInput |
                        BufferUsageFlags::ShaderDeviceAddress,
                    ResourceMemoryUsage::CpuToGpu, true,
                    "Vulkan shared RT instances");
        }
        m_prepared = true; m_recorded = false;
        m_statistics.sceneGeneration = scene.statistics().generation;
        m_statistics.lastUpdate = scene.lastUpdateKind();
        m_statistics.blasCount = static_cast<uint32_t>(m_blas.size());
        m_statistics.instanceCount = count;
        m_statistics.resultBytes = resultBytes + sizes.accelerationStructureSize;
        m_statistics.scratchBytes = m_scratch.size;
        m_statistics.state = RayTracingAccelerationStructureState::BuildPending;
        return true;
    }

    void VulkanAccelerationStructureProvider::recordBuild(VkCommandBuffer cmd,
        const RayTracingSceneService& scene,
        const std::unordered_map<AssetHandle, VulkanUploadedModel,
            AssetHandleHash>& models, uint32_t frameSlot)
    {
        if (!m_prepared || !cmd || frameSlot >= m_instanceBuffers.size()) return;
        const auto geometries = scene.geometries();
        const bool buildBlas = !m_blasValid ||
            scene.lastUpdateKind() == RayTracingSceneUpdateKind::GeometryRebuild;
        if (buildBlas)
        {
            for (size_t i = 0; i < geometries.size(); ++i)
            {
                const VulkanUploadedModel& model = models.at(geometries[i].model);
                VkAccelerationStructureGeometryTrianglesDataKHR triangles{
                    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR };
                triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
                triangles.vertexData.deviceAddress = model.vertexBuffer.gpuAddress;
                triangles.vertexStride = sizeof(AssetVertex);
                triangles.maxVertex = static_cast<uint32_t>(model.vertexBuffer.size /
                    sizeof(AssetVertex));
                triangles.indexType = VK_INDEX_TYPE_UINT32;
                triangles.indexData.deviceAddress = model.indexBuffer.gpuAddress +
                    VkDeviceSize(geometries[i].firstIndex) * sizeof(uint32_t);
                VkAccelerationStructureGeometryKHR geometry{
                    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
                geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
                geometry.flags = nativeGeometryFlags(geometries[i].flags);
                geometry.geometry.triangles = triangles;
                VkAccelerationStructureBuildGeometryInfoKHR build{
                    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
                build.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
                build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                    VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
                build.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
                build.dstAccelerationStructure = m_blas[i].as;
                build.geometryCount = 1; build.pGeometries = &geometry;
                build.scratchData.deviceAddress = m_scratch.gpuAddress;
                VkAccelerationStructureBuildRangeInfoKHR range{};
                range.primitiveCount = geometries[i].indexCount / 3u;
                const VkAccelerationStructureBuildRangeInfoKHR* ranges[] = { &range };
                m_cmdBuild(cmd, 1, &build, ranges);
                VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                barrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
                barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
                barrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
                barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                    VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
                VkDependencyInfo dependency{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                dependency.memoryBarrierCount = 1; dependency.pMemoryBarriers = &barrier;
                vkCmdPipelineBarrier2(cmd, &dependency);
            }
            m_statistics.blasBuildCount += geometries.size();
            m_blasValid = true;
        }
        else m_statistics.blasReuseCount += geometries.size();

        VulkanBuffer& instanceBuffer = m_instanceBuffers[frameSlot];
        auto* native = static_cast<VkAccelerationStructureInstanceKHR*>(instanceBuffer.mapped);
        for (size_t i = 0; i < scene.instances().size(); ++i)
        {
            const auto& source = scene.instances()[i];
            VkAccelerationStructureDeviceAddressInfoKHR addressInfo{
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR };
            addressInfo.accelerationStructure = m_blas[source.geometryIndex].as;
            VkAccelerationStructureInstanceKHR instance{};
            instance.transform = nativeTransform(source.world);
            instance.instanceCustomIndex = source.instanceId & 0x00ffffffu;
            instance.mask = source.mask & 0xffu;
            instance.instanceShaderBindingTableRecordOffset =
                source.geometryIndex & 0x00ffffffu;
            instance.flags = (geometries[source.geometryIndex].flags &
                RayTracingGeometryDoubleSided) != 0
                ? VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR : 0u;
            instance.accelerationStructureReference =
                m_getAddress(m_device, &addressInfo);
            native[i] = instance;
        }

        VkAccelerationStructureGeometryInstancesDataKHR instances{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR };
        instances.data.deviceAddress = instanceBuffer.gpuAddress;
        VkAccelerationStructureGeometryKHR geometry{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
        geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        geometry.geometry.instances = instances;
        const bool refit = scene.lastUpdateKind() ==
            RayTracingSceneUpdateKind::TlasRefit && m_hasTlas;
        VkAccelerationStructureBuildGeometryInfoKHR build{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
        build.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
            VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
        build.mode = refit ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR
                           : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        build.srcAccelerationStructure = refit ? m_tlas : VK_NULL_HANDLE;
        build.dstAccelerationStructure = m_tlas;
        build.geometryCount = 1; build.pGeometries = &geometry;
        build.scratchData.deviceAddress = m_scratch.gpuAddress;
        VkAccelerationStructureBuildRangeInfoKHR range{};
        range.primitiveCount = static_cast<uint32_t>(scene.instances().size());
        const VkAccelerationStructureBuildRangeInfoKHR* ranges[] = { &range };
        m_cmdBuild(cmd, 1, &build, ranges);
        VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        VkDependencyInfo dependency{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dependency.memoryBarrierCount = 1; dependency.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dependency);
        if (refit) ++m_statistics.tlasRefitCount;
        else ++m_statistics.tlasBuildCount;
        m_hasTlas = true; m_recorded = true; m_prepared = false;
        m_statistics.state = RayTracingAccelerationStructureState::Ready;
        spdlog::info(
            "[Vulkan RTAS] ready generation={} BLAS={} instances={} update={}",
            m_statistics.sceneGeneration, m_statistics.blasCount,
            m_statistics.instanceCount,
            static_cast<uint32_t>(m_statistics.lastUpdate));
    }
}
