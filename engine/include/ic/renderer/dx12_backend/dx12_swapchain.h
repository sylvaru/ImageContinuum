#pragma once

#include <array>
#include <cstdint>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

namespace ic
{
    class DX12Device;
    class DX12Factory;
    class Window;

    enum class DX12SwapchainState : uint8_t
    {
        Valid,
        Minimized,
        OutOfDate
    };

    class DX12Swapchain
    {
    public:
        static constexpr uint32_t MaxImages = 3;

        void init(
            const DX12Factory& factory,
            const DX12Device& device,
            Window& window,
            uint32_t requestedImageCount);

        void shutdown();
        void resize();

        bool present();

        bool updateSizeFromWindow();

        ID3D12Resource* currentBackBuffer() const;
        D3D12_CPU_DESCRIPTOR_HANDLE currentRtv() const;

        uint32_t currentBackBufferIndex() const
        {
            return m_currentBackBufferIndex;
        }

        uint32_t imageCount() const
        {
            return m_imageCount;
        }

        uint32_t width() const
        {
            return m_width;
        }

        uint32_t height() const
        {
            return m_height;
        }

        DXGI_FORMAT format() const
        {
            return m_format;
        }

        DX12SwapchainState state() const
        {
            return m_state;
        }

        bool validForRendering() const
        {
            return m_state == DX12SwapchainState::Valid;
        }

        bool vsyncEnabled() const
        {
            return m_vsync;
        }

        void setVsyncEnabled(bool enabled)
        {
            m_vsync = enabled;
        }

    private:
        void createSwapchain();
        void createRenderTargets();
        void releaseRenderTargets();
        void queryWindowSize(uint32_t& width, uint32_t& height) const;

        const DX12Factory* m_factory = nullptr;
        const DX12Device* m_device = nullptr;
        Window* m_window = nullptr;

        Microsoft::WRL::ComPtr<IDXGISwapChain4> m_swapchain;
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_rtvHeap;
        std::array<Microsoft::WRL::ComPtr<ID3D12Resource>, MaxImages> m_backBuffers;

        uint32_t m_imageCount = 2;
        uint32_t m_currentBackBufferIndex = 0;
        uint32_t m_width = 0;
        uint32_t m_height = 0;
        bool m_vsync = true;

        DXGI_FORMAT m_format = DXGI_FORMAT_R8G8B8A8_UNORM;
        DX12SwapchainState m_state = DX12SwapchainState::OutOfDate;
    };
}
