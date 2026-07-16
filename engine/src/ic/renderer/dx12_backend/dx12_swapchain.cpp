#include "ic/renderer/dx12_backend/dx12_swapchain.h"

#include "ic/interface/window.h"
#include "ic/renderer/dx12_backend/dx12_device.h"
#include "ic/renderer/dx12_backend/dx12_factory.h"

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <spdlog/spdlog.h>

#include <algorithm>
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

    HWND hwndFromWindow(ic::Window& window)
    {
        auto* glfwWindow = static_cast<GLFWwindow*>(window.getNativeHandle());
        if (!glfwWindow)
        {
            throw std::runtime_error("DX12 swapchain requires a GLFW window handle.");
        }

        HWND hwnd = glfwGetWin32Window(glfwWindow);
        if (!hwnd)
        {
            throw std::runtime_error("Failed to query Win32 HWND from GLFW window.");
        }

        return hwnd;
    }
}

namespace ic
{
    void DX12Swapchain::init(
        const DX12Factory& factory,
        const DX12Device& device,
        Window& window,
        uint32_t requestedImageCount)
    {
        m_factory = &factory;
        m_device = &device;
        m_window = &window;
        m_imageCount = std::clamp(requestedImageCount, 2u, MaxImages);

        queryWindowSize(m_width, m_height);
        if (m_width == 0 || m_height == 0)
        {
            m_state = DX12SwapchainState::Minimized;
            spdlog::warn("[DX12Swapchain] Window is minimized; delaying swapchain creation.");
            return;
        }

        createSwapchain();
        createRenderTargets();

        m_state = DX12SwapchainState::Valid;
        spdlog::info(
            "[DX12Swapchain] Created ({} images, {}x{})",
            m_imageCount,
            m_width,
            m_height);
    }

    void DX12Swapchain::shutdown()
    {
        releaseRenderTargets();
        m_rtvHeap.Reset();
        m_swapchain.Reset();
        m_factory = nullptr;
        m_device = nullptr;
        m_window = nullptr;
        m_width = 0;
        m_height = 0;
        m_currentBackBufferIndex = 0;
        m_state = DX12SwapchainState::OutOfDate;
        spdlog::info("[DX12Swapchain] Shutdown");
    }

    void DX12Swapchain::resize()
    {
        if (!m_factory || !m_device || !m_window)
        {
            return;
        }

        uint32_t newWidth = 0;
        uint32_t newHeight = 0;
        queryWindowSize(newWidth, newHeight);

        if (newWidth == 0 || newHeight == 0)
        {
            releaseRenderTargets();
            m_state = DX12SwapchainState::Minimized;
            return;
        }

        m_width = newWidth;
        m_height = newHeight;

        if (!m_swapchain)
        {
            createSwapchain();
        }
        else
        {
            releaseRenderTargets();

            throwIfFailed(
                m_swapchain->ResizeBuffers(
                    m_imageCount,
                    m_width,
                    m_height,
                    m_format,
                    m_allowTearing
                        ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING
                        : 0u),
                "Failed to resize DX12 swapchain buffers.");
        }

        createRenderTargets();
        m_currentBackBufferIndex = m_swapchain->GetCurrentBackBufferIndex();
        m_state = DX12SwapchainState::Valid;

        spdlog::info(
            "[DX12Swapchain] Resized ({}x{})",
            m_width,
            m_height);
    }

    bool DX12Swapchain::present()
    {
        if (!m_swapchain || m_state != DX12SwapchainState::Valid)
        {
            return false;
        }

        // VSync on: sync to vblank (syncInterval 1). VSync off: present
        // immediately with the ALLOW_TEARING flag so the display is not locked to
        // the refresh rate. The tearing flag is only legal with syncInterval 0
        // and in windowed mode (we never take exclusive fullscreen).
        const UINT syncInterval = m_vsync ? 1u : 0u;
        const UINT flags =
            (!m_vsync && m_allowTearing) ? DXGI_PRESENT_ALLOW_TEARING : 0u;

        HRESULT hr = m_swapchain->Present(syncInterval, flags);
        if (hr == DXGI_ERROR_DEVICE_REMOVED ||
            hr == DXGI_ERROR_DEVICE_RESET ||
            hr == DXGI_STATUS_OCCLUDED)
        {
            m_state = DX12SwapchainState::OutOfDate;
            return false;
        }

        throwIfFailed(hr, "Failed to present DX12 swapchain.");

        m_currentBackBufferIndex = m_swapchain->GetCurrentBackBufferIndex();
        return true;
    }

    bool DX12Swapchain::updateSizeFromWindow()
    {
        uint32_t newWidth = 0;
        uint32_t newHeight = 0;
        queryWindowSize(newWidth, newHeight);

        if (newWidth == 0 || newHeight == 0)
        {
            if (m_state != DX12SwapchainState::Minimized)
            {
                m_state = DX12SwapchainState::OutOfDate;
            }
            return false;
        }

        if (newWidth != m_width || newHeight != m_height || m_state != DX12SwapchainState::Valid)
        {
            m_state = DX12SwapchainState::OutOfDate;
            return false;
        }

        return true;
    }

    ID3D12Resource* DX12Swapchain::currentBackBuffer() const
    {
        return m_backBuffers[m_currentBackBufferIndex].Get();
    }

    D3D12_CPU_DESCRIPTOR_HANDLE DX12Swapchain::currentRtv() const
    {
        D3D12_CPU_DESCRIPTOR_HANDLE handle =
            m_rtvHeap->GetCPUDescriptorHandleForHeapStart();

        handle.ptr +=
            static_cast<SIZE_T>(m_currentBackBufferIndex) *
            m_device->rtvDescriptorSize();

        return handle;
    }

    void DX12Swapchain::createSwapchain()
    {
        DXGI_SWAP_CHAIN_DESC1 desc{};
        desc.Width = m_width;
        desc.Height = m_height;
        desc.Format = m_format;
        desc.Stereo = FALSE;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        desc.BufferCount = m_imageCount;
        desc.Scaling = DXGI_SCALING_STRETCH;
        desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
        // Enable tearing on the swapchain when the adapter/OS supports it, so a
        // non-vsynced present can run uncapped (see present()). The same flag
        // must be carried through ResizeBuffers.
        m_allowTearing = m_factory->allowTearing();
        desc.Flags = m_allowTearing
            ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING
            : 0u;

        Microsoft::WRL::ComPtr<IDXGISwapChain1> swapchain1;
        throwIfFailed(
            m_factory->factory()->CreateSwapChainForHwnd(
                m_device->graphicsQueue(),
                hwndFromWindow(*m_window),
                &desc,
                nullptr,
                nullptr,
                &swapchain1),
            "Failed to create DX12 swapchain.");

        throwIfFailed(
            m_factory->factory()->MakeWindowAssociation(
                hwndFromWindow(*m_window),
                DXGI_MWA_NO_ALT_ENTER),
            "Failed to configure DXGI window association.");

        throwIfFailed(
            swapchain1.As(&m_swapchain),
            "Failed to query IDXGISwapChain4.");

        m_currentBackBufferIndex = m_swapchain->GetCurrentBackBufferIndex();

        // Report the refresh rate of the monitor the window is on. With VSync
        // enabled the frame rate is capped to exactly this value (windowed
        // present syncs to the compositor at the display's current mode), so it
        // is the first thing to check when "VSync fps" looks lower than the
        // monitor's advertised maximum. The mode may not be set to that maximum.
        HMONITOR monitor = MonitorFromWindow(
            hwndFromWindow(*m_window), MONITOR_DEFAULTTONEAREST);
        MONITORINFOEXW monitorInfo{};
        monitorInfo.cbSize = sizeof(monitorInfo);
        DEVMODEW mode{};
        mode.dmSize = sizeof(mode);
        if (GetMonitorInfoW(monitor, &monitorInfo) &&
            EnumDisplaySettingsW(
                monitorInfo.szDevice, ENUM_CURRENT_SETTINGS, &mode))
        {
            spdlog::info(
                "[DX12Swapchain] Output mode {}x{} @ {} Hz "
                "(VSync caps FPS to this refresh rate)",
                mode.dmPelsWidth, mode.dmPelsHeight, mode.dmDisplayFrequency);
        }
    }

    void DX12Swapchain::createRenderTargets()
    {
        D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        heapDesc.NumDescriptors = m_imageCount;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        heapDesc.NodeMask = 0;

        throwIfFailed(
            m_device->device()->CreateDescriptorHeap(
                &heapDesc,
                IID_PPV_ARGS(&m_rtvHeap)),
            "Failed to create DX12 RTV descriptor heap.");

        D3D12_CPU_DESCRIPTOR_HANDLE handle =
            m_rtvHeap->GetCPUDescriptorHandleForHeapStart();

        for (uint32_t i = 0; i < m_imageCount; ++i)
        {
            throwIfFailed(
                m_swapchain->GetBuffer(i, IID_PPV_ARGS(&m_backBuffers[i])),
                "Failed to query DX12 swapchain back buffer.");

            m_device->device()->CreateRenderTargetView(
                m_backBuffers[i].Get(),
                nullptr,
                handle);

            handle.ptr += m_device->rtvDescriptorSize();
        }
    }

    void DX12Swapchain::releaseRenderTargets()
    {
        for (Microsoft::WRL::ComPtr<ID3D12Resource>& buffer : m_backBuffers)
        {
            buffer.Reset();
        }
        m_rtvHeap.Reset();
    }

    void DX12Swapchain::queryWindowSize(uint32_t& width, uint32_t& height) const
    {
        auto* glfwWindow = static_cast<GLFWwindow*>(m_window->getNativeHandle());

        int framebufferWidth = 0;
        int framebufferHeight = 0;
        glfwGetFramebufferSize(
            glfwWindow,
            &framebufferWidth,
            &framebufferHeight);

        width = static_cast<uint32_t>(std::max(framebufferWidth, 0));
        height = static_cast<uint32_t>(std::max(framebufferHeight, 0));
    }
}
