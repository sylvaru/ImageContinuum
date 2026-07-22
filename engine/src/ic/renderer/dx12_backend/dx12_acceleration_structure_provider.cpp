#include "ic/common/ic_pch.h"
#include "ic/renderer/dx12_backend/dx12_acceleration_structure_provider.h"

#include "ic/core/asset_manager.h"

#include <algorithm>
#include <cstring>
#include <spdlog/spdlog.h>

namespace ic
{
    namespace
    {
        uint64_t aligned(uint64_t value, uint64_t alignment)
        {
            return (value + alignment - 1u) & ~(alignment - 1u);
        }

        D3D12_RAYTRACING_GEOMETRY_FLAGS geometryFlags(uint32_t flags)
        {
            return (flags & RayTracingGeometryOpaque) != 0
                ? D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE
                : D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
        }
    }

    void DX12AccelerationStructureProvider::init(
        const DX12Device& device,
        DX12ResourceAllocator& allocator,
        DX12RetirementQueue& retirement,
        uint32_t framesInFlight)
    {
        m_device = device.device();
        m_allocator = &allocator;
        m_retirement = &retirement;
        m_tier = device.features().rayTracingTier;
        m_instanceBuffers.resize(std::max(1u, framesInFlight));
        m_statistics.state =
            m_tier >= D3D12_RAYTRACING_TIER_1_1
                ? RayTracingAccelerationStructureState::Empty
                : RayTracingAccelerationStructureState::Unsupported;
    }

    void DX12AccelerationStructureProvider::shutdown()
    {
        if (!m_allocator)
            return;
        for (Blas& blas : m_blas)
            m_allocator->destroyBuffer(blas.result);
        m_blas.clear();
        m_allocator->destroyBuffer(m_scratch);
        m_allocator->destroyBuffer(m_tlas);
        for (DX12Buffer& buffer : m_instanceBuffers)
            m_allocator->destroyBuffer(buffer);
        m_instanceBuffers.clear();
        m_device = nullptr;
        m_allocator = nullptr;
        m_retirement = nullptr;
    }

    void DX12AccelerationStructureProvider::retire(DX12Buffer& buffer)
    {
        if (!buffer)
            return;
        m_retirement->retire(std::move(buffer));
        ++m_statistics.retiredResourceCount;
    }

    void DX12AccelerationStructureProvider::ensureBuffer(
        DX12Buffer& buffer,
        uint64_t size,
        BufferUsageFlags usage,
        ResourceMemoryUsage memory,
        bool mapped,
        const char* name)
    {
        size = std::max<uint64_t>(size, 256u);
        if (buffer && buffer.size >= size)
            return;
        retire(buffer);
        buffer = m_allocator->createBuffer({
            .size = size,
            .usage = usage,
            .memoryUsage = memory,
            .mappedAtCreation = mapped,
            .debugName = name });
    }

    void DX12AccelerationStructureProvider::setEnabled(bool enabled)
    {
        if (m_enabled == enabled)
            return;
        m_enabled = enabled;
        invalidate();
        m_statistics.state = enabled
            ? (m_tier >= D3D12_RAYTRACING_TIER_1_1
                ? RayTracingAccelerationStructureState::Empty
                : RayTracingAccelerationStructureState::Unsupported)
            : RayTracingAccelerationStructureState::Disabled;
    }

    void DX12AccelerationStructureProvider::invalidate() noexcept
    {
        m_prepared = false;
        m_recorded = false;
        m_hasTlas = false;
        m_blasValid = false;
        m_statistics.sceneGeneration = 0;
        if (m_enabled)
            m_statistics.state = RayTracingAccelerationStructureState::Empty;
    }

    RayTracingCapabilities
        DX12AccelerationStructureProvider::capabilities() const noexcept
    {
        const bool hardware =
            m_tier != D3D12_RAYTRACING_TIER_NOT_SUPPORTED;
        return {
            .accelerationStructures = hardware,
            .inlineRayQueries = m_tier >= D3D12_RAYTRACING_TIER_1_1,
            .accelerationStructureUpdates = hardware,
            .indirectDispatch = true,
            .sharedAccelerationStructures =
                m_enabled && m_recorded &&
                m_statistics.state ==
                    RayTracingAccelerationStructureState::Ready };
    }

    bool DX12AccelerationStructureProvider::readyFor(
        uint64_t sceneGeneration) const noexcept
    {
        return capabilities().sharedAccelerationStructures &&
            m_statistics.sceneGeneration == sceneGeneration;
    }

    uint64_t DX12AccelerationStructureProvider::shaderTlasHandle()
        const noexcept
    {
        return readyFor(m_statistics.sceneGeneration)
            ? m_tlas.gpuAddress : 0;
    }

    bool DX12AccelerationStructureProvider::prepare(
        const RayTracingSceneService& scene,
        const std::unordered_map<
            AssetHandle, DX12UploadedModel, AssetHandleHash>& models)
    {
        if ((m_prepared || m_recorded) &&
            m_statistics.sceneGeneration == scene.statistics().generation)
            return m_prepared;
        if (!m_enabled || m_tier < D3D12_RAYTRACING_TIER_1_1 ||
            !scene.statistics().valid || scene.instances().empty())
        {
            m_recorded = false;
            m_prepared = false;
            m_statistics.state = m_enabled
                ? RayTracingAccelerationStructureState::Empty
                : RayTracingAccelerationStructureState::Disabled;
            return false;
        }

        const auto geometries = scene.geometries();
        if (geometries.empty())
            return false;

        const bool geometryRebuild =
            scene.lastUpdateKind() ==
                RayTracingSceneUpdateKind::GeometryRebuild ||
            m_blas.size() != geometries.size();
        if (geometryRebuild)
        {
            for (Blas& blas : m_blas)
                retire(blas.result);
            m_blas.clear();
            m_blas.resize(geometries.size());
            m_blasValid = false;
        }

        uint64_t maxScratch = 0;
        uint64_t resultBytes = 0;
        for (size_t i = 0; i < geometries.size(); ++i)
        {
            const RayTracingGeometryRecord& geometry = geometries[i];
            const auto modelIt = models.find(geometry.model);
            if (modelIt == models.end() || !modelIt->second.uploaded)
                return false;
            const DX12UploadedModel& model = modelIt->second;

            D3D12_RAYTRACING_GEOMETRY_DESC nativeGeometry{};
            nativeGeometry.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
            nativeGeometry.Flags = geometryFlags(geometry.flags);
            nativeGeometry.Triangles.VertexBuffer.StartAddress =
                model.vertexBuffer.gpuAddress;
            nativeGeometry.Triangles.VertexBuffer.StrideInBytes =
                sizeof(AssetVertex);
            nativeGeometry.Triangles.VertexCount = static_cast<UINT>(
                model.vertexBuffer.size / sizeof(AssetVertex));
            nativeGeometry.Triangles.VertexFormat =
                DXGI_FORMAT_R32G32B32_FLOAT;
            nativeGeometry.Triangles.IndexBuffer =
                model.indexBuffer.gpuAddress +
                static_cast<uint64_t>(geometry.firstIndex) * sizeof(uint32_t);
            nativeGeometry.Triangles.IndexCount = geometry.indexCount;
            nativeGeometry.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;

            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs{};
            inputs.Type =
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
            inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
            inputs.NumDescs = 1;
            inputs.pGeometryDescs = &nativeGeometry;
            inputs.Flags =
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE |
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;

            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info{};
            m_device->GetRaytracingAccelerationStructurePrebuildInfo(
                &inputs, &info);
            if (info.ResultDataMaxSizeInBytes == 0)
                return false;
            Blas& blas = m_blas[i];
            blas.model = geometry.model;
            blas.resultSize = aligned(info.ResultDataMaxSizeInBytes, 256u);
            ensureBuffer(
                blas.result,
                blas.resultSize,
                BufferUsageFlags::AccelerationStructureStorage |
                    BufferUsageFlags::ShaderDeviceAddress,
                ResourceMemoryUsage::GpuOnly,
                false,
                "DX12 shared RT BLAS");
            maxScratch = std::max<uint64_t>(
                maxScratch, info.ScratchDataSizeInBytes);
            resultBytes += blas.resultSize;
        }

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasInputs{};
        tlasInputs.Type =
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        tlasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        tlasInputs.NumDescs =
            static_cast<UINT>(scene.instances().size());
        tlasInputs.Flags =
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE |
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO tlasInfo{};
        m_device->GetRaytracingAccelerationStructurePrebuildInfo(
            &tlasInputs, &tlasInfo);
        m_tlasResultSize = aligned(tlasInfo.ResultDataMaxSizeInBytes, 256u);
        m_tlasScratchSize = aligned(std::max(
            tlasInfo.ScratchDataSizeInBytes,
            tlasInfo.UpdateScratchDataSizeInBytes), 256u);
        ensureBuffer(
            m_tlas, m_tlasResultSize,
            BufferUsageFlags::AccelerationStructureStorage |
                BufferUsageFlags::ShaderDeviceAddress,
            ResourceMemoryUsage::GpuOnly, false, "DX12 shared RT TLAS");
        ensureBuffer(
            m_scratch, std::max(maxScratch, m_tlasScratchSize),
            BufferUsageFlags::Storage |
                BufferUsageFlags::AccelerationStructureScratch |
                BufferUsageFlags::ShaderDeviceAddress,
            ResourceMemoryUsage::GpuOnly, false, "DX12 shared RT scratch");

        const uint32_t neededInstances =
            static_cast<uint32_t>(scene.instances().size());
        if (neededInstances > m_instanceCapacity)
        {
            m_instanceCapacity = std::max(
                neededInstances,
                std::max(64u, m_instanceCapacity * 2u));
            for (DX12Buffer& buffer : m_instanceBuffers)
            {
                ensureBuffer(
                    buffer,
                    static_cast<uint64_t>(m_instanceCapacity) *
                        sizeof(D3D12_RAYTRACING_INSTANCE_DESC),
                    BufferUsageFlags::Storage |
                        BufferUsageFlags::ShaderDeviceAddress,
                    ResourceMemoryUsage::CpuToGpu,
                    true,
                    "DX12 shared RT instances");
            }
        }

        m_prepared = true;
        m_recorded = false;
        m_statistics.sceneGeneration = scene.statistics().generation;
        m_statistics.lastUpdate = scene.lastUpdateKind();
        m_statistics.blasCount = static_cast<uint32_t>(m_blas.size());
        m_statistics.instanceCount = neededInstances;
        m_statistics.resultBytes = resultBytes + m_tlasResultSize;
        m_statistics.scratchBytes = m_scratch.size;
        m_statistics.state =
            RayTracingAccelerationStructureState::BuildPending;
        return true;
    }

    void DX12AccelerationStructureProvider::recordBuild(
        ID3D12GraphicsCommandList4* cmd,
        const RayTracingSceneService& scene,
        const std::unordered_map<
            AssetHandle, DX12UploadedModel, AssetHandleHash>& models,
        uint32_t frameSlot)
    {
        if (!m_prepared || !cmd || frameSlot >= m_instanceBuffers.size())
            return;

        const auto geometries = scene.geometries();
        const bool buildBlas =
            !m_blasValid || scene.lastUpdateKind() ==
                RayTracingSceneUpdateKind::GeometryRebuild;
        if (buildBlas)
        {
            std::unordered_set<AssetHandle, AssetHandleHash> transitioned;
            for (size_t i = 0; i < geometries.size(); ++i)
            {
                const RayTracingGeometryRecord& geometry = geometries[i];
                const DX12UploadedModel& model = models.at(geometry.model);
                if (transitioned.insert(geometry.model).second)
                {
                    D3D12_RESOURCE_BARRIER barriers[2]{};
                    for (uint32_t b = 0; b < 2; ++b)
                    {
                        barriers[b].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                        barriers[b].Transition.Subresource =
                            D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                    }
                    barriers[0].Transition.pResource =
                        model.vertexBuffer.resource.Get();
                    barriers[0].Transition.StateBefore =
                        D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
                    barriers[0].Transition.StateAfter =
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                    barriers[1].Transition.pResource =
                        model.indexBuffer.resource.Get();
                    barriers[1].Transition.StateBefore =
                        D3D12_RESOURCE_STATE_INDEX_BUFFER;
                    barriers[1].Transition.StateAfter =
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                    cmd->ResourceBarrier(2, barriers);
                }

                D3D12_RAYTRACING_GEOMETRY_DESC nativeGeometry{};
                nativeGeometry.Type =
                    D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
                nativeGeometry.Flags = geometryFlags(geometry.flags);
                nativeGeometry.Triangles.VertexBuffer.StartAddress =
                    model.vertexBuffer.gpuAddress;
                nativeGeometry.Triangles.VertexBuffer.StrideInBytes =
                    sizeof(AssetVertex);
                nativeGeometry.Triangles.VertexCount = static_cast<UINT>(
                    model.vertexBuffer.size / sizeof(AssetVertex));
                nativeGeometry.Triangles.VertexFormat =
                    DXGI_FORMAT_R32G32B32_FLOAT;
                nativeGeometry.Triangles.IndexBuffer =
                    model.indexBuffer.gpuAddress +
                    static_cast<uint64_t>(geometry.firstIndex) *
                        sizeof(uint32_t);
                nativeGeometry.Triangles.IndexCount = geometry.indexCount;
                nativeGeometry.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;

                D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build{};
                build.Inputs.Type =
                    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
                build.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
                build.Inputs.NumDescs = 1;
                build.Inputs.pGeometryDescs = &nativeGeometry;
                build.Inputs.Flags =
                    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE |
                    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;
                build.ScratchAccelerationStructureData = m_scratch.gpuAddress;
                build.DestAccelerationStructureData =
                    m_blas[i].result.gpuAddress;
                cmd->BuildRaytracingAccelerationStructure(
                    &build, 0, nullptr);
                D3D12_RESOURCE_BARRIER uav{};
                uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                uav.UAV.pResource = m_blas[i].result.resource.Get();
                cmd->ResourceBarrier(1, &uav);
            }
            for (AssetHandle handle : transitioned)
            {
                const DX12UploadedModel& model = models.at(handle);
                D3D12_RESOURCE_BARRIER barriers[2]{};
                for (uint32_t b = 0; b < 2; ++b)
                {
                    barriers[b].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                    barriers[b].Transition.Subresource =
                        D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                }
                barriers[0].Transition.pResource =
                    model.vertexBuffer.resource.Get();
                barriers[0].Transition.StateBefore =
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                barriers[0].Transition.StateAfter =
                    D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
                barriers[1].Transition.pResource =
                    model.indexBuffer.resource.Get();
                barriers[1].Transition.StateBefore =
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                barriers[1].Transition.StateAfter =
                    D3D12_RESOURCE_STATE_INDEX_BUFFER;
                cmd->ResourceBarrier(2, barriers);
            }
            m_statistics.blasBuildCount += geometries.size();
            m_blasValid = true;
        }
        else
        {
            m_statistics.blasReuseCount += geometries.size();
        }

        DX12Buffer& instanceBuffer = m_instanceBuffers[frameSlot];
        auto* nativeInstances = static_cast<D3D12_RAYTRACING_INSTANCE_DESC*>(
            instanceBuffer.mapped);
        for (size_t i = 0; i < scene.instances().size(); ++i)
        {
            const RayTracingInstanceRecord& instance = scene.instances()[i];
            D3D12_RAYTRACING_INSTANCE_DESC native{};
            for (uint32_t row = 0; row < 3; ++row)
                for (uint32_t column = 0; column < 4; ++column)
                    native.Transform[row][column] =
                        instance.world[column][row];
            native.InstanceID = instance.instanceId & 0x00ffffffu;
            native.InstanceContributionToHitGroupIndex =
                instance.geometryIndex & 0x00ffffffu;
            native.InstanceMask = static_cast<UINT8>(instance.mask);
            const uint32_t geometryFlagsValue =
                geometries[instance.geometryIndex].flags;
            native.Flags =
                (geometryFlagsValue & RayTracingGeometryDoubleSided) != 0
                    ? D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE
                    : D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
            native.AccelerationStructure =
                m_blas[instance.geometryIndex].result.gpuAddress;
            nativeInstances[i] = native;
        }

        const bool refit =
            scene.lastUpdateKind() == RayTracingSceneUpdateKind::TlasRefit &&
            m_hasTlas;
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC tlasBuild{};
        tlasBuild.Inputs.Type =
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        tlasBuild.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        tlasBuild.Inputs.NumDescs =
            static_cast<UINT>(scene.instances().size());
        tlasBuild.Inputs.InstanceDescs = instanceBuffer.gpuAddress;
        tlasBuild.Inputs.Flags =
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE |
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
        if (refit)
        {
            tlasBuild.Inputs.Flags |=
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
            tlasBuild.SourceAccelerationStructureData = m_tlas.gpuAddress;
        }
        tlasBuild.ScratchAccelerationStructureData = m_scratch.gpuAddress;
        tlasBuild.DestAccelerationStructureData = m_tlas.gpuAddress;
        cmd->BuildRaytracingAccelerationStructure(&tlasBuild, 0, nullptr);
        D3D12_RESOURCE_BARRIER tlasBarrier{};
        tlasBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        tlasBarrier.UAV.pResource = m_tlas.resource.Get();
        cmd->ResourceBarrier(1, &tlasBarrier);

        if (refit)
            ++m_statistics.tlasRefitCount;
        else
            ++m_statistics.tlasBuildCount;
        m_recorded = true;
        m_prepared = false;
        m_hasTlas = true;
        m_statistics.state = RayTracingAccelerationStructureState::Ready;
        spdlog::info(
            "[DX12 RTAS] ready generation={} BLAS={} instances={} update={}",
            m_statistics.sceneGeneration, m_statistics.blasCount,
            m_statistics.instanceCount,
            static_cast<uint32_t>(m_statistics.lastUpdate));
    }
}
