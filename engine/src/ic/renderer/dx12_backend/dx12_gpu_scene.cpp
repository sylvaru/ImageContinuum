#include "ic/renderer/dx12_backend/dx12_gpu_scene.h"

#include "ic/renderer/dx12_backend/dx12_device.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ic
{
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
    }

    void DX12GpuScene::init(
        const DX12Device& device,
        DX12ResourceAllocator& resourceAllocator,
        DX12DescriptorSystem& descriptorSystem,
        uint32_t framesInFlight)
    {
        m_device = device.device();
        m_resourceAllocator = &resourceAllocator;
        m_descriptorSystem = &descriptorSystem;
        m_frames.clear();
        m_frames.resize(std::max(1u, framesInFlight));
        m_prepared = PreparedGpuScene{};
    }

    void DX12GpuScene::shutdown()
    {
        destroyCullBuffers();

        for (DX12GpuSceneFrameResources& frame : m_frames)
        {
            m_resourceAllocator->destroyBuffer(frame.frameConstants);
            m_resourceAllocator->destroyBuffer(frame.objects);
            m_resourceAllocator->destroyBuffer(frame.materials);
            m_resourceAllocator->destroyBuffer(frame.visibleLights);
            m_resourceAllocator->destroyBuffer(frame.instanceBounds);
            m_resourceAllocator->destroyBuffer(frame.drawInputs);
            m_descriptorSystem->releaseResourceDescriptors(frame.objectSrv);
            m_descriptorSystem->releaseResourceDescriptors(frame.materialSrv);
        }
        m_frames.clear();

        indexedIndirectCommandSignature.Reset();

        m_device = nullptr;
        m_resourceAllocator = nullptr;
        m_descriptorSystem = nullptr;
    }

    void DX12GpuScene::ensureCullBuffers(
        ID3D12Device5* device,
        uint32_t maxCullInstances,
        uint32_t maxBins)
    {
        destroyCullBuffers();

        maxInstances = maxCullInstances;

        visibleInstances = m_resourceAllocator->createBuffer({
            .size = maxCullInstances * sizeof(uint32_t),
            .usage = BufferUsageFlags::Storage,
            .memoryUsage = ResourceMemoryUsage::GpuOnly,
            .debugName = "DX12 GPU cull visible instances"
        });
        visibleInstanceCount = m_resourceAllocator->createBuffer({
            .size = sizeof(uint32_t),
            .usage = BufferUsageFlags::Storage | BufferUsageFlags::TransferSrc,
            .memoryUsage = ResourceMemoryUsage::GpuOnly,
            .debugName = "DX12 GPU cull visible count"
        });
        visibleInstanceCountReadback = m_resourceAllocator->createBuffer({
            .size = sizeof(uint32_t),
            .usage = BufferUsageFlags::TransferDst,
            .memoryUsage = ResourceMemoryUsage::GpuToCpu,
            .mappedAtCreation = true,
            .debugName = "DX12 GPU cull visible count readback"
        });
        indirectArguments = m_resourceAllocator->createBuffer({
            .size = maxCullInstances * sizeof(GpuIndexedIndirectArguments),
            .usage = BufferUsageFlags::Storage | BufferUsageFlags::Indirect,
            .memoryUsage = ResourceMemoryUsage::GpuOnly,
            .debugName = "DX12 GPU-driven indirect arguments"
        });
        drawMetadata = m_resourceAllocator->createBuffer({
            .size = maxCullInstances * sizeof(GpuDrawMetadata),
            .usage = BufferUsageFlags::Storage,
            .memoryUsage = ResourceMemoryUsage::GpuOnly,
            .debugName = "DX12 GPU-driven draw metadata"
        });
        binCounts = m_resourceAllocator->createBuffer({
            .size = maxBins * sizeof(uint32_t),
            .usage = BufferUsageFlags::Storage | BufferUsageFlags::Indirect,
            .memoryUsage = ResourceMemoryUsage::GpuOnly,
            .debugName = "DX12 GPU-driven bin counts"
        });

        visibleInstancesState = visibleInstances.initialState;
        visibleInstanceCountState = visibleInstanceCount.initialState;
        indirectArgumentsState = indirectArguments.initialState;
        drawMetadataState = drawMetadata.initialState;
        binCountsState = binCounts.initialState;

        if (!indexedIndirectCommandSignature)
        {
            D3D12_INDIRECT_ARGUMENT_DESC argument{};
            argument.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;
            D3D12_COMMAND_SIGNATURE_DESC signature{};
            signature.ByteStride = sizeof(GpuIndexedIndirectArguments);
            signature.NumArgumentDescs = 1;
            signature.pArgumentDescs = &argument;
            throwIfFailed(
                device->CreateCommandSignature(
                    &signature, nullptr,
                    IID_PPV_ARGS(&indexedIndirectCommandSignature)),
                "Failed to create DX12 indexed indirect command signature.");
        }
    }

    void DX12GpuScene::destroyCullBuffers()
    {
        m_resourceAllocator->destroyBuffer(visibleInstances);
        m_resourceAllocator->destroyBuffer(visibleInstanceCount);
        m_resourceAllocator->destroyBuffer(visibleInstanceCountReadback);
        m_resourceAllocator->destroyBuffer(indirectArguments);
        m_resourceAllocator->destroyBuffer(drawMetadata);
        m_resourceAllocator->destroyBuffer(binCounts);
        visibleInstancesState = D3D12_RESOURCE_STATE_COMMON;
        visibleInstanceCountState = D3D12_RESOURCE_STATE_COMMON;
        indirectArgumentsState = D3D12_RESOURCE_STATE_COMMON;
        drawMetadataState = D3D12_RESOURCE_STATE_COMMON;
        binCountsState = D3D12_RESOURCE_STATE_COMMON;
    }

    bool DX12GpuScene::prepare(
        uint64_t frameIndex,
        const SceneRenderView& scene,
        uint32_t frameSlot,
        const ResolveModelFn& resolveModel,
        const BuildFrameDataFn& buildFrameData)
    {
        if (frameSlot >= m_frames.size())
        {
            return false;
        }

        const bool alreadyPreparedThisFrame =
            m_prepared.valid() && m_prepared.frameIndex() == frameIndex;

        if (!m_prepared.build(frameIndex, scene, resolveModel))
        {
            return false;
        }

        if (alreadyPreparedThisFrame)
        {
            return true;
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

        DX12GpuSceneFrameResources& frameResources = m_frames[frameSlot];

        constexpr uint64_t constantBufferAlignment = 256;
        const uint64_t frameSize =
            (sizeof(GpuFrameData) + constantBufferAlignment - 1u) &
            ~(constantBufferAlignment - 1u);

        if (!frameResources.frameConstants)
        {
            frameResources.frameConstants =
                m_resourceAllocator->createBuffer({
                    .size = frameSize,
                    .usage = BufferUsageFlags::Constant,
                    .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                    .mappedAtCreation = true,
                    .debugName = "DX12 frame constants"
                });
        }

        if (objects.size() > frameResources.objectCapacity)
        {
            m_resourceAllocator->destroyBuffer(frameResources.objects);
            m_descriptorSystem->releaseResourceDescriptors(
                frameResources.objectSrv);
            frameResources.objectSrv = {};

            frameResources.objects =
                m_resourceAllocator->createBuffer({
                    .size = objects.size() * sizeof(GpuObjectData),
                    .usage = BufferUsageFlags::None,
                    .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                    .mappedAtCreation = true,
                    .debugName = "DX12 object data"
                });
            frameResources.objectCapacity =
                static_cast<uint32_t>(objects.size());

            frameResources.objectSrv =
                m_descriptorSystem->allocateResourceDescriptors(1);

            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.Shader4ComponentMapping =
                D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            srv.Format = DXGI_FORMAT_UNKNOWN;
            srv.Buffer.NumElements = frameResources.objectCapacity;
            srv.Buffer.StructureByteStride = sizeof(GpuObjectData);

            m_device->CreateShaderResourceView(
                frameResources.objects.resource.Get(),
                &srv,
                frameResources.objectSrv.cpuStart);
        }

        if (materials.size() > frameResources.materialCapacity)
        {
            m_resourceAllocator->destroyBuffer(frameResources.materials);
            m_descriptorSystem->releaseResourceDescriptors(
                frameResources.materialSrv);
            frameResources.materialSrv = {};

            frameResources.materials =
                m_resourceAllocator->createBuffer({
                    .size = materials.size() * sizeof(GpuMaterialData),
                    .usage = BufferUsageFlags::None,
                    .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                    .mappedAtCreation = true,
                    .debugName = "DX12 material data"
                });
            frameResources.materialCapacity =
                static_cast<uint32_t>(materials.size());

            frameResources.materialSrv =
                m_descriptorSystem->allocateResourceDescriptors(1);

            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.Shader4ComponentMapping =
                D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            srv.Format = DXGI_FORMAT_UNKNOWN;
            srv.Buffer.NumElements = frameResources.materialCapacity;
            srv.Buffer.StructureByteStride = sizeof(GpuMaterialData);

            m_device->CreateShaderResourceView(
                frameResources.materials.resource.Get(),
                &srv,
                frameResources.materialSrv.cpuStart);
        }

        if (visibleLights.size() > frameResources.visibleLightCapacity)
        {
            m_resourceAllocator->destroyBuffer(frameResources.visibleLights);
            frameResources.visibleLights =
                m_resourceAllocator->createBuffer({
                    .size = visibleLights.size() * sizeof(GpuVisibleLight),
                    .usage = BufferUsageFlags::None,
                    .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                    .mappedAtCreation = true,
                    .debugName = "DX12 clustered visible lights"
                });
            frameResources.visibleLightCapacity =
                static_cast<uint32_t>(visibleLights.size());
        }

        if (instanceBounds.size() > frameResources.instanceBoundsCapacity)
        {
            m_resourceAllocator->destroyBuffer(frameResources.instanceBounds);
            frameResources.instanceBounds =
                m_resourceAllocator->createBuffer({
                    .size = instanceBounds.size() * sizeof(GpuInstanceBounds),
                    .usage = BufferUsageFlags::None,
                    .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                    .mappedAtCreation = true,
                    .debugName = "DX12 GPU cull instance bounds"
                });
            frameResources.instanceBoundsCapacity =
                static_cast<uint32_t>(instanceBounds.size());
        }
        if (drawInputs.size() > frameResources.drawInputCapacity)
        {
            m_resourceAllocator->destroyBuffer(frameResources.drawInputs);
            frameResources.drawInputs = m_resourceAllocator->createBuffer({
                .size = drawInputs.size() * sizeof(GpuDrawInput),
                .usage = BufferUsageFlags::None,
                .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                .mappedAtCreation = true,
                .debugName = "DX12 GPU-driven draw inputs" });
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

        return true;
    }
}
