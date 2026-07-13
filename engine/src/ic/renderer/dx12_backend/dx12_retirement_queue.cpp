#include "ic/renderer/dx12_backend/dx12_retirement_queue.h"

#include <algorithm>
#include <stdexcept>

namespace ic
{
    void DX12RetirementQueue::init(
        DX12ResourceAllocator& allocator,
        DX12DescriptorSystem& descriptors,
        uint32_t framesInFlight)
    {
        m_allocator = &allocator;
        m_descriptors = &descriptors;
        m_slots.resize(std::max(1u, framesInFlight));
    }

    void DX12RetirementQueue::beginFrame(uint32_t frameSlot)
    {
        std::scoped_lock lock(m_mutex);
        if (frameSlot >= m_slots.size())
        {
            throw std::out_of_range("DX12 retirement frame slot is out of range.");
        }
        m_currentSlot = frameSlot;
        recycle(m_slots[frameSlot]);
    }

    void DX12RetirementQueue::retire(DX12Buffer&& buffer)
    {
        std::scoped_lock lock(m_mutex);
        if (buffer)
        {
            m_slots[m_currentSlot].buffers.emplace_back(std::move(buffer));
        }
    }

    void DX12RetirementQueue::retire(DX12DescriptorAllocation allocation)
    {
        std::scoped_lock lock(m_mutex);
        if (allocation.valid())
        {
            m_slots[m_currentSlot].descriptors.push_back(allocation);
        }
    }

    void DX12RetirementQueue::drain()
    {
        std::scoped_lock lock(m_mutex);
        for (Slot& slot : m_slots)
        {
            recycle(slot);
        }
        m_slots.clear();
        m_allocator = nullptr;
        m_descriptors = nullptr;
    }

    void DX12RetirementQueue::recycle(Slot& slot)
    {
        for (DX12Buffer& buffer : slot.buffers)
        {
            m_allocator->destroyBuffer(buffer);
        }
        for (DX12DescriptorAllocation allocation : slot.descriptors)
        {
            m_descriptors->releaseResourceDescriptors(allocation);
        }
        slot.buffers.clear();
        slot.descriptors.clear();
    }
}
