#include "ic/renderer/dx12_backend/dx12_command_system.h"

#include <spdlog/spdlog.h>

#include <stdexcept>

namespace
{
    void throwIfFailed(HRESULT hr, const char* message)
    {
        if (FAILED(hr))
        {
            throw std::runtime_error(message);
        }
    }

    uint32_t queueIndex(ic::QueueType queue)
    {
        return static_cast<uint32_t>(queue);
    }

    D3D12_COMMAND_LIST_TYPE commandListType(ic::QueueType queue)
    {
        switch (queue)
        {
        case ic::QueueType::Compute: return D3D12_COMMAND_LIST_TYPE_COMPUTE;
        case ic::QueueType::Transfer: return D3D12_COMMAND_LIST_TYPE_COPY;
        case ic::QueueType::Graphics: return D3D12_COMMAND_LIST_TYPE_DIRECT;
        }
        return D3D12_COMMAND_LIST_TYPE_DIRECT;
    }
}

namespace ic
{
    void DX12CommandSystem::init(
        ID3D12Device* device,
        uint32_t framesInFlight,
        uint32_t workerCount)
    {
        if (!device)
        {
            throw std::runtime_error("DX12CommandSystem requires a valid device.");
        }

        m_device = device;

        const uint32_t frameCount = framesInFlight == 0 ? 1 : framesInFlight;
        const uint32_t workers = workerCount == 0 ? 1 : workerCount;

        m_frames.resize(frameCount);

        for (FrameCommands& frame : m_frames)
        {
            for (uint32_t queue = 0; queue < frame.queues.size(); ++queue)
            {
                auto& queueWorkers = frame.queues[queue];
                queueWorkers.resize(workers);
                for (std::unique_ptr<WorkerCommands>& workerPtr : queueWorkers)
                {
                    workerPtr = std::make_unique<WorkerCommands>();
                    WorkerCommands& worker = *workerPtr;

                    worker.slots.emplace_back(std::make_unique<CommandSlot>());
                    CommandSlot& slot = *worker.slots.back();
                    const auto type = commandListType(
                        static_cast<QueueType>(queue));

                    throwIfFailed(
                        m_device->CreateCommandAllocator(
                            type, IID_PPV_ARGS(&slot.allocator)),
                        "Failed to create D3D12 command allocator.");

                    throwIfFailed(
                        m_device->CreateCommandList(
                            0, type, slot.allocator.Get(), nullptr,
                            IID_PPV_ARGS(&slot.list)),
                        "Failed to create D3D12 command list.");

                    throwIfFailed(
                        slot.list->Close(),
                        "Failed to close initial D3D12 command list.");
                }
            }
        }

        spdlog::info(
            "[DX12CommandSystem] Initialized (frames={}, workers={})",
            frameCount,
            workers);
    }

    void DX12CommandSystem::shutdown()
    {
        m_frames.clear();
        m_device = nullptr;
        spdlog::info("[DX12CommandSystem] Shutdown");
    }

    void DX12CommandSystem::beginFrame(uint32_t frameIndex)
    {
        if (frameIndex >= m_frames.size())
        {
            throw std::runtime_error("DX12 frame index out of range.");
        }

        FrameCommands& frame = m_frames[frameIndex];

        for (auto& queueWorkers : frame.queues)
        for (const std::unique_ptr<WorkerCommands>& workerPtr : queueWorkers)
        {
            WorkerCommands& worker = *workerPtr;
            std::scoped_lock lock(worker.mutex);
            worker.nextSlot = 0;

            for (const std::unique_ptr<CommandSlot>& slot : worker.slots)
            {
                throwIfFailed(
                    slot->allocator->Reset(),
                    "Failed to reset D3D12 command allocator.");
            }
        }
    }

    DX12CommandSystem::RecordingLease
        DX12CommandSystem::acquireFrameCommandList(
            uint32_t frameIndex,
            uint32_t workerIndex,
            QueueType queue)
    {
        if (frameIndex >= m_frames.size())
        {
            throw std::runtime_error("DX12 frame index out of range.");
        }

        FrameCommands& frame = m_frames[frameIndex];

        auto& workers = frame.queues[queueIndex(queue)];
        if (workerIndex >= workers.size())
        {
            throw std::runtime_error("DX12 worker index out of range.");
        }

        WorkerCommands& worker = *workers[workerIndex];
        std::unique_lock lock(worker.mutex);

        if (worker.nextSlot >= worker.slots.size())
        {
            auto slot = std::make_unique<CommandSlot>();

            const auto type = commandListType(queue);
            throwIfFailed(
                m_device->CreateCommandAllocator(
                    type,
                    IID_PPV_ARGS(&slot->allocator)),
                "Failed to create D3D12 command allocator.");

            throwIfFailed(
                m_device->CreateCommandList(
                    0,
                    type,
                    slot->allocator.Get(),
                    nullptr,
                    IID_PPV_ARGS(&slot->list)),
                "Failed to create D3D12 command list.");

            throwIfFailed(
                slot->list->Close(),
                "Failed to close initial D3D12 command list.");

            worker.slots.push_back(std::move(slot));
        }

        CommandSlot& slot =
            *worker.slots[worker.nextSlot++];

        throwIfFailed(
            slot.list->Reset(slot.allocator.Get(), nullptr),
            "Failed to reset D3D12 command list.");

        return RecordingLease(slot.list.Get(), std::move(lock));
    }

    ID3D12GraphicsCommandList4* DX12CommandSystem::beginFrameCommandList(
        uint32_t frameIndex,
        uint32_t workerIndex)
    {
        if (frameIndex >= m_frames.size())
        {
            throw std::runtime_error("DX12 frame index out of range.");
        }

        FrameCommands& frame = m_frames[frameIndex];

        auto& workers = frame.queues[queueIndex(QueueType::Graphics)];
        if (workerIndex >= workers.size())
        {
            throw std::runtime_error("DX12 worker index out of range.");
        }

        WorkerCommands& worker = *workers[workerIndex];
        std::scoped_lock lock(worker.mutex);

        if (worker.slots.empty())
        {
            throw std::runtime_error("DX12 worker has no command slots.");
        }

        CommandSlot& slot = *worker.slots[0];

        throwIfFailed(
            slot.list->Reset(slot.allocator.Get(), nullptr),
            "Failed to reset D3D12 command list.");

        return slot.list.Get();
    }
}
