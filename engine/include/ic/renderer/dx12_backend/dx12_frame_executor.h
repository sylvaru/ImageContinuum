#pragma once

#include "ic/renderer/dx12_backend/dx12_device.h"
#include "ic/renderer/dx12_backend/dx12_swapchain.h"
#include "ic/renderer/frame_graph/compiled_graph_plan.h"

#include <cstdint>
#include <vector>

#include <d3d12.h>
#include <wrl/client.h>

namespace ic
{
    class DX12FrameExecutor final
    {
    public:
        DX12FrameExecutor() = default;
        ~DX12FrameExecutor() 
        {
            shutdown();
        }
        DX12FrameExecutor(const DX12FrameExecutor&) = delete;
        DX12FrameExecutor& operator=(const DX12FrameExecutor&) = delete;

        DX12FrameExecutor(DX12FrameExecutor&&) = delete;
        DX12FrameExecutor& operator=(DX12FrameExecutor&&) = delete;

        void init(
            const DX12Device& device,
            DX12Swapchain& swapchain,
            uint32_t framesInFlight);
        void shutdown();

        [[nodiscard]] uint32_t framesInFlight() const noexcept
        {
            return static_cast<uint32_t>(m_frameSync.size());
        }

        [[nodiscard]] bool ready() const noexcept
        {
            return !m_frameSync.empty();
        }

        // Blocks until the GPU work previously submitted for this slot has
        // completed.
        void waitForFrame(uint32_t frameSlot);

        // Signals the graphics queue and records the value for this slot.
        void signalFrame(uint32_t frameSlot);

        // Fully drains the GPU (used before teardown / swapchain recreation).
        void waitForGpu();

        // Submits the recorded command lists honoring the compiled queue
        // schedule, presents, and signals this frame's fence. Returns whether
        // present succeeded.
        bool submitAndPresent(
            const CompiledGraphPlan& plan,
            const std::vector<ID3D12CommandList*>& commandLists,
            uint32_t frameSlot);

    private:
        struct FrameSync
        {
            uint64_t fenceValue = 0;
        };

        const DX12Device* m_device = nullptr;
        DX12Swapchain* m_swapchain = nullptr;
        Microsoft::WRL::ComPtr<ID3D12Fence> m_fence;
        HANDLE m_fenceEvent = nullptr;
        uint64_t m_nextFenceValue = 1;
        uint64_t m_lastGraphCompletionValue = 0;
        std::vector<FrameSync> m_frameSync;
    };
}
