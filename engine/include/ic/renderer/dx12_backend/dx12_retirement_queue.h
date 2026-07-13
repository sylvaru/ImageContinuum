#pragma once

#include "ic/renderer/dx12_backend/dx12_descriptor_system.h"
#include "ic/renderer/dx12_backend/dx12_resource_allocator.h"

#include <cstdint>
#include <mutex>
#include <vector>

namespace ic
{
    // Backend-local frame-slot retirement. beginFrame is called only after the
    // frame executor has observed completion for that slot, so recycling adds
    // no CPU or GPU-wide wait.
    class DX12RetirementQueue final
    {
    public:
        void init(
            DX12ResourceAllocator& allocator,
            DX12DescriptorSystem& descriptors,
            uint32_t framesInFlight);
        void beginFrame(uint32_t frameSlot);
        void retire(DX12Buffer&& buffer);
        void retire(DX12DescriptorAllocation allocation);
        void drain();

    private:
        struct Slot
        {
            std::vector<DX12Buffer> buffers;
            std::vector<DX12DescriptorAllocation> descriptors;
        };

        void recycle(Slot& slot);

        DX12ResourceAllocator* m_allocator = nullptr;
        DX12DescriptorSystem* m_descriptors = nullptr;
        std::vector<Slot> m_slots;
        uint32_t m_currentSlot = 0;
        std::mutex m_mutex;
    };
}
