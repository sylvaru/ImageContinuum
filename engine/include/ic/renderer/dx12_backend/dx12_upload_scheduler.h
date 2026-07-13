#pragma once

#include "ic/renderer/dx12_backend/dx12_device.h"
#include "ic/renderer/dx12_backend/dx12_resource_allocator.h"

#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

#include <d3d12.h>
#include <wrl/client.h>

namespace ic
{
    struct DX12UploadDependency
    {
        ID3D12Fence* fence = nullptr;
        uint64_t value = 0;
        ID3D12CommandQueue* queue = nullptr;

        [[nodiscard]] bool valid() const noexcept
        {
            return fence != nullptr && value != 0 && queue != nullptr;
        }
    };

    // Batches same-frame runtime copies on the direct queue. This preserves
    // immediate graphics ordering while the upload fence supplies the required
    // dependency to asynchronous compute. Dedicated copy is only profitable
    // once residency can be staged ahead of first use.
    // The frame executor joins the returned fence into every consumer queue
    // and the graphics frame fence, making slot-based staging retirement safe.
    class DX12UploadScheduler final
    {
    public:
        void init(
            const DX12Device& device,
            DX12ResourceAllocator& allocator,
            uint32_t framesInFlight);
        void shutdown();
        void beginFrame(uint32_t frameSlot);
        void record(
            const std::function<void(ID3D12GraphicsCommandList4*)>& commands);
        void retire(DX12Buffer&& buffer);
        [[nodiscard]] DX12UploadDependency flush();

    private:
        struct BatchCommand
        {
            Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
            Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> commandList;
        };

        struct Slot
        {
            std::vector<BatchCommand> commands;
            std::vector<DX12Buffer> retiredBuffers;
            uint32_t activeCommandCount = 0;
        };

        const DX12Device* m_device = nullptr;
        DX12ResourceAllocator* m_allocator = nullptr;
        Microsoft::WRL::ComPtr<ID3D12Fence> m_fence;
        ID3D12CommandQueue* m_queue = nullptr;
        std::vector<Slot> m_slots;
        uint64_t m_nextFenceValue = 1;
        uint32_t m_currentSlot = 0;
        std::mutex m_mutex;
    };
}
