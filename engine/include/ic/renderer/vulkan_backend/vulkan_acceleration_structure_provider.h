#pragma once

#include <unordered_map>
#include <vector>

#include "ic/renderer/ray_tracing/ray_tracing_acceleration_structure.h"
#include "ic/renderer/vulkan_backend/vulkan_gpu_scene.h"
#include "ic/renderer/vulkan_backend/vulkan_device.h"
#include "ic/renderer/vulkan_backend/vulkan_retirement_queue.h"

namespace ic
{
    class VulkanAccelerationStructureProvider final
        : public RayTracingAccelerationStructureProvider
    {
    public:
        void init(const VulkanDevice& device,
            VulkanResourceAllocator& allocator,
            VulkanRetirementQueue& retirement, uint32_t framesInFlight);
        void shutdown();
        bool prepare(const RayTracingSceneService& scene,
            const std::unordered_map<AssetHandle, VulkanUploadedModel,
                AssetHandleHash>& models);
        void recordBuild(VkCommandBuffer cmd,
            const RayTracingSceneService& scene,
            const std::unordered_map<AssetHandle, VulkanUploadedModel,
                AssetHandleHash>& models, uint32_t frameSlot);

        void setEnabled(bool enabled) override;
        [[nodiscard]] bool enabled() const noexcept override { return m_enabled; }
        void invalidate() noexcept override;
        [[nodiscard]] RayTracingCapabilities capabilities() const noexcept override;
        [[nodiscard]] bool readyFor(uint64_t generation) const noexcept override;
        [[nodiscard]] uint64_t shaderTlasHandle() const noexcept override;
        [[nodiscard]] VkAccelerationStructureKHR nativeTlas() const noexcept
        { return m_recorded ? m_tlas : VK_NULL_HANDLE; }
        [[nodiscard]] const RayTracingAccelerationStructureStatistics&
            statistics() const noexcept override { return m_statistics; }

    private:
        struct Blas { VulkanBuffer storage; VkAccelerationStructureKHR as = VK_NULL_HANDLE; };
        void retire(VulkanBuffer& buffer);
        void retire(VkAccelerationStructureKHR& as);
        void ensureBuffer(VulkanBuffer& buffer, VkDeviceSize size,
            BufferUsageFlags usage, ResourceMemoryUsage memory,
            bool mapped, const char* name);
        VkAccelerationStructureKHR createAs(
            VkAccelerationStructureTypeKHR type,
            const VulkanBuffer& storage, VkDeviceSize size);

        VkDevice m_device = VK_NULL_HANDLE;
        VulkanResourceAllocator* m_allocator = nullptr;
        VulkanRetirementQueue* m_retirement = nullptr;
        bool m_hardware = false;
        bool m_enabled = true;
        bool m_prepared = false;
        bool m_recorded = false;
        bool m_blasValid = false;
        bool m_hasTlas = false;
        std::vector<Blas> m_blas;
        VulkanBuffer m_scratch;
        VulkanBuffer m_tlasStorage;
        VkAccelerationStructureKHR m_tlas = VK_NULL_HANDLE;
        std::vector<VulkanBuffer> m_instanceBuffers;
        uint32_t m_instanceCapacity = 0;
        RayTracingAccelerationStructureStatistics m_statistics{};
        PFN_vkCreateAccelerationStructureKHR m_createAs = nullptr;
        PFN_vkDestroyAccelerationStructureKHR m_destroyAs = nullptr;
        PFN_vkGetAccelerationStructureBuildSizesKHR m_getBuildSizes = nullptr;
        PFN_vkCmdBuildAccelerationStructuresKHR m_cmdBuild = nullptr;
        PFN_vkGetAccelerationStructureDeviceAddressKHR m_getAddress = nullptr;
    };
}
