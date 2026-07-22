#pragma once

#include <unordered_map>
#include <vector>

#include "ic/renderer/dx12_backend/dx12_gpu_scene.h"
#include "ic/renderer/dx12_backend/dx12_retirement_queue.h"
#include "ic/renderer/ray_tracing/ray_tracing_acceleration_structure.h"

namespace ic
{
    class DX12AccelerationStructureProvider final
        : public RayTracingAccelerationStructureProvider
    {
    public:
        void init(
            const DX12Device& device,
            DX12ResourceAllocator& allocator,
            DX12RetirementQueue& retirement,
            uint32_t framesInFlight);
        void shutdown();

        bool prepare(
            const RayTracingSceneService& scene,
            const std::unordered_map<
                AssetHandle, DX12UploadedModel, AssetHandleHash>& models);
        void recordBuild(
            ID3D12GraphicsCommandList4* cmd,
            const RayTracingSceneService& scene,
            const std::unordered_map<
                AssetHandle, DX12UploadedModel, AssetHandleHash>& models,
            uint32_t frameSlot);

        void setEnabled(bool enabled) override;
        [[nodiscard]] bool enabled() const noexcept override
        {
            return m_enabled;
        }
        void invalidate() noexcept override;
        [[nodiscard]] RayTracingCapabilities capabilities()
            const noexcept override;
        [[nodiscard]] bool readyFor(
            uint64_t sceneGeneration) const noexcept override;
        [[nodiscard]] uint64_t shaderTlasHandle() const noexcept override;
        [[nodiscard]] const RayTracingAccelerationStructureStatistics&
            statistics() const noexcept override
        {
            return m_statistics;
        }

    private:
        struct Blas
        {
            AssetHandle model = {};
            DX12Buffer result;
            uint64_t resultSize = 0;
        };

        void retire(DX12Buffer& buffer);
        void ensureBuffer(
            DX12Buffer& buffer,
            uint64_t size,
            BufferUsageFlags usage,
            ResourceMemoryUsage memory,
            bool mapped,
            const char* name);

        ID3D12Device5* m_device = nullptr;
        DX12ResourceAllocator* m_allocator = nullptr;
        DX12RetirementQueue* m_retirement = nullptr;
        D3D12_RAYTRACING_TIER m_tier =
            D3D12_RAYTRACING_TIER_NOT_SUPPORTED;
        std::vector<Blas> m_blas;
        DX12Buffer m_scratch;
        DX12Buffer m_tlas;
        std::vector<DX12Buffer> m_instanceBuffers;
        uint64_t m_tlasResultSize = 0;
        uint64_t m_tlasScratchSize = 0;
        uint32_t m_instanceCapacity = 0;
        bool m_enabled = true;
        bool m_prepared = false;
        bool m_recorded = false;
        bool m_hasTlas = false;
        bool m_blasValid = false;
        RayTracingAccelerationStructureStatistics m_statistics{};
    };
}
