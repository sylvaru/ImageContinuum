#pragma once

#include <memory>
#include <mutex>
#include <array>
#include <vector>
#include "ic/renderer/frame_graph/frame_graph_types.h"

#include <d3d12.h>
#include <wrl/client.h>

namespace ic
{
    class DX12CommandSystem
    {
    public:
        void init(
            ID3D12Device* device,
            uint32_t framesInFlight,
            uint32_t workerCount);

        void shutdown();
        void beginFrame(uint32_t frameIndex);

        class RecordingLease
        {
        public:
            RecordingLease() = default;
            RecordingLease(const RecordingLease&) = delete;
            RecordingLease& operator=(const RecordingLease&) = delete;
            RecordingLease(RecordingLease&&) noexcept = default;
            RecordingLease& operator=(RecordingLease&&) noexcept = default;

            ID3D12GraphicsCommandList4* commandList() const
            {
                return m_commandList;
            }

            explicit operator bool() const
            {
                return m_commandList != nullptr;
            }

        private:
            friend class DX12CommandSystem;

            RecordingLease(
                ID3D12GraphicsCommandList4* commandList,
                std::unique_lock<std::mutex>&& lock)
                : m_commandList(commandList)
                , m_lock(std::move(lock))
            {
            }

            ID3D12GraphicsCommandList4* m_commandList = nullptr;
            std::unique_lock<std::mutex> m_lock;
        };

        RecordingLease acquireFrameCommandList(
            uint32_t frameIndex,
            uint32_t workerIndex,
            QueueType queue = QueueType::Graphics);

    private:
        struct CommandSlot
        {
            Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
            Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> list;
        };

        struct WorkerCommands
        {
            std::vector<std::unique_ptr<CommandSlot>> slots;
            uint32_t nextSlot = 0;
            std::mutex mutex;
        };

        struct FrameCommands
        {
            std::array<
                std::vector<std::unique_ptr<WorkerCommands>>, 3> queues;
        };

        ID3D12Device* m_device = nullptr;
        std::vector<FrameCommands> m_frames;
    };
}
