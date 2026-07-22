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
        m_indirectCommandSignatures.clear();
        m_frames.resize(std::max(1u, framesInFlight));
        m_prepared = PreparedGpuScene{};
    }

    void DX12GpuScene::shutdown()
    {
        for (DX12GpuSceneFrameResources& frame : m_frames)
        {
            m_resourceAllocator->destroyBuffer(frame.frameConstants);
            m_resourceAllocator->destroyBuffer(frame.objects);
            m_resourceAllocator->destroyBuffer(frame.materials);
            m_resourceAllocator->destroyBuffer(frame.giRtGeometries);
            m_resourceAllocator->destroyBuffer(frame.giRtInstances);
            m_resourceAllocator->destroyBuffer(frame.giTraceConstants);
            m_descriptorSystem->releaseResourceDescriptors(frame.objectSrv);
            m_descriptorSystem->releaseResourceDescriptors(frame.materialSrv);
            m_descriptorSystem->releaseResourceDescriptors(
                frame.giModelBufferSrvs);
        }
        m_frames.clear();

        m_device = nullptr;
        m_resourceAllocator = nullptr;
        m_descriptorSystem = nullptr;
    }

    ID3D12CommandSignature* DX12GpuScene::indirectCommandSignature(
        ID3D12RootSignature* rootSignature)
    {
        if (!rootSignature || !m_device)
        {
            return nullptr;
        }
        auto existing = m_indirectCommandSignatures.find(rootSignature);
        if (existing != m_indirectCommandSignatures.end())
        {
            return existing->second.Get();
        }

        D3D12_INDIRECT_ARGUMENT_DESC arguments[2]{};
        arguments[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
        arguments[0].Constant.RootParameterIndex = 3;
        arguments[0].Constant.Num32BitValuesToSet = 4;
        arguments[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;

        D3D12_COMMAND_SIGNATURE_DESC desc{};
        desc.ByteStride = sizeof(DX12GpuIndexedIndirectCommand);
        desc.NumArgumentDescs = static_cast<UINT>(std::size(arguments));
        desc.pArgumentDescs = arguments;

        Microsoft::WRL::ComPtr<ID3D12CommandSignature> signature;
        throwIfFailed(
            m_device->CreateCommandSignature(
                &desc, rootSignature, IID_PPV_ARGS(&signature)),
            "Failed to create DX12 GPU-driven command signature.");
        ID3D12CommandSignature* result = signature.Get();
        m_indirectCommandSignatures.emplace(rootSignature, std::move(signature));
        return result;
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
        // Only the count is needed here (for GpuFrameData); the bounds/inputs
        // themselves are uploaded by the backend into the graph-owned buffers.
        const std::span<const GpuInstanceBounds> instanceBounds =
            m_prepared.instanceBounds();

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

        return true;
    }
}
