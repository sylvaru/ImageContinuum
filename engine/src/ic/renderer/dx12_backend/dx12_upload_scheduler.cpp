#include "ic/renderer/dx12_backend/dx12_upload_scheduler.h"

#include <algorithm>
#include <spdlog/spdlog.h>
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

    void DX12UploadScheduler::init(
        const DX12Device& device,
        DX12ResourceAllocator& allocator,
        uint32_t framesInFlight)
    {
        m_device = &device;
        m_allocator = &allocator;
        m_queue = device.graphicsQueue();
        m_slots.resize(std::max(1u, framesInFlight));

        throwIfFailed(
            device.device()->CreateFence(
                0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)),
            "Failed to create DX12 upload fence.");

        spdlog::info(
            "[DX12UploadScheduler] Initialized (queue=direct, frames={})",
            m_slots.size());
    }

    void DX12UploadScheduler::shutdown()
    {
        std::scoped_lock lock(m_mutex);
        for (Slot& slot : m_slots)
        {
            for (DX12Buffer& buffer : slot.retiredBuffers)
            {
                m_allocator->destroyBuffer(buffer);
            }
            slot.retiredBuffers.clear();
            slot.commands.clear();
        }
        m_slots.clear();
        m_fence.Reset();
        m_nextFenceValue = 1;
        m_currentSlot = 0;
        m_queue = nullptr;
        m_allocator = nullptr;
        m_device = nullptr;
    }

    void DX12UploadScheduler::beginFrame(uint32_t frameSlot)
    {
        std::scoped_lock lock(m_mutex);
        if (frameSlot >= m_slots.size())
        {
            throw std::out_of_range("DX12 upload frame slot is out of range.");
        }

        m_currentSlot = frameSlot;
        Slot& slot = m_slots[frameSlot];
        for (DX12Buffer& buffer : slot.retiredBuffers)
        {
            m_allocator->destroyBuffer(buffer);
        }
        slot.retiredBuffers.clear();

        slot.activeCommandCount = 0;
    }

    void DX12UploadScheduler::record(
        const std::function<void(ID3D12GraphicsCommandList4*)>& commands)
    {
        std::scoped_lock lock(m_mutex);
        Slot& slot = m_slots[m_currentSlot];
        if (slot.activeCommandCount == slot.commands.size())
        {
            BatchCommand command{};
            throwIfFailed(
                m_device->device()->CreateCommandAllocator(
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    IID_PPV_ARGS(&command.allocator)),
                "Failed to create pooled DX12 upload allocator.");
            throwIfFailed(
                m_device->device()->CreateCommandList(
                    0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                    command.allocator.Get(), nullptr,
                    IID_PPV_ARGS(&command.commandList)),
                "Failed to create pooled DX12 upload command list.");
            throwIfFailed(
                command.commandList->Close(),
                "Failed to close pooled DX12 upload command list.");
            slot.commands.emplace_back(std::move(command));
        }
        BatchCommand& command = slot.commands[slot.activeCommandCount++];
        throwIfFailed(
            command.allocator->Reset(),
            "Failed to reset pooled DX12 upload allocator.");
        throwIfFailed(
            command.commandList->Reset(command.allocator.Get(), nullptr),
            "Failed to reset pooled DX12 upload command list.");
        commands(command.commandList.Get());
        throwIfFailed(
            command.commandList->Close(),
            "Failed to close pooled DX12 upload command list.");
    }

    void DX12UploadScheduler::retire(DX12Buffer&& buffer)
    {
        std::scoped_lock lock(m_mutex);
        if (buffer)
        {
            m_slots[m_currentSlot].retiredBuffers.emplace_back(
                std::move(buffer));
        }
    }

    DX12UploadDependency DX12UploadScheduler::flush()
    {
        std::scoped_lock lock(m_mutex);
        Slot& slot = m_slots[m_currentSlot];
        if (slot.activeCommandCount == 0)
        {
            return {};
        }

        std::vector<ID3D12CommandList*> lists;
        lists.reserve(slot.activeCommandCount);
        for (uint32_t i = 0; i < slot.activeCommandCount; ++i)
        {
            lists.push_back(slot.commands[i].commandList.Get());
        }
        m_queue->ExecuteCommandLists(
            static_cast<UINT>(lists.size()), lists.data());
        const uint64_t value = m_nextFenceValue++;
        throwIfFailed(
            m_queue->Signal(m_fence.Get(), value),
            "Failed to signal DX12 upload fence.");
        return { m_fence.Get(), value, m_queue };
    }
}
