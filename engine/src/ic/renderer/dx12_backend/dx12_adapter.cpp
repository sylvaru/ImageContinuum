#include "ic/renderer/dx12_backend/dx12_adapter.h"

#include "ic/renderer/dx12_backend/dx12_factory.h"

#include <d3d12.h>
#include <spdlog/spdlog.h>
#include <Windows.h>

#include <stdexcept>
#include <string>

namespace
{
    std::string narrow(const wchar_t* text)
    {
        if (!text || text[0] == L'\0')
        {
            return {};
        }

        const int required = WideCharToMultiByte(
            CP_UTF8,
            0,
            text,
            -1,
            nullptr,
            0,
            nullptr,
            nullptr);

        if (required <= 1)
        {
            return {};
        }

        std::string result(static_cast<size_t>(required), '\0');

        WideCharToMultiByte(
            CP_UTF8,
            0,
            text,
            -1,
            result.data(),
            required,
            nullptr,
            nullptr);

        result.pop_back();
        return result;
    }
}

namespace ic
{
    void DX12Adapter::init(const DX12Factory& factory)
    {
        IDXGIFactory6* dxgiFactory = factory.factory();
        if (!dxgiFactory)
        {
            throw std::runtime_error("DX12 adapter selection requires a valid DXGI factory.");
        }

        spdlog::info("[DX12Adapter] Selecting adapter...");

        for (UINT index = 0;; ++index)
        {
            Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter1;
            HRESULT hr = dxgiFactory->EnumAdapterByGpuPreference(
                index,
                DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                IID_PPV_ARGS(&adapter1));

            if (hr == DXGI_ERROR_NOT_FOUND)
            {
                break;
            }

            if (FAILED(hr))
            {
                continue;
            }

            Microsoft::WRL::ComPtr<IDXGIAdapter4> adapter4;
            if (SUCCEEDED(adapter1.As(&adapter4)) && isAdapterSuitable(adapter4.Get()))
            {
                m_adapter = adapter4;
                break;
            }
        }

        if (!m_adapter)
        {
            for (UINT index = 0;; ++index)
            {
                Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter1;
                HRESULT hr = dxgiFactory->EnumAdapters1(index, &adapter1);

                if (hr == DXGI_ERROR_NOT_FOUND)
                {
                    break;
                }

                if (FAILED(hr))
                {
                    continue;
                }

                Microsoft::WRL::ComPtr<IDXGIAdapter4> adapter4;
                if (SUCCEEDED(adapter1.As(&adapter4)) && isAdapterSuitable(adapter4.Get()))
                {
                    m_adapter = adapter4;
                    break;
                }
            }
        }

        if (!m_adapter)
        {
            throw std::runtime_error("No suitable DX12 adapter found.");
        }

        m_adapter->GetDesc3(&m_desc);
        logAdapterInfo();
    }

    void DX12Adapter::shutdown()
    {
        m_adapter.Reset();
        m_desc = {};
    }

    bool DX12Adapter::isAdapterSuitable(IDXGIAdapter4* adapter) const
    {
        DXGI_ADAPTER_DESC3 desc{};
        adapter->GetDesc3(&desc);

        if ((desc.Flags & DXGI_ADAPTER_FLAG3_SOFTWARE) != 0)
        {
            return false;
        }

        return SUCCEEDED(D3D12CreateDevice(
            adapter,
            D3D_FEATURE_LEVEL_12_0,
            __uuidof(ID3D12Device),
            nullptr));
    }

    void DX12Adapter::logAdapterInfo() const
    {
        const uint64_t dedicatedMb =
            m_desc.DedicatedVideoMemory / (1024ull * 1024ull);

        spdlog::info("----------- DX12 Adapter -----------");
        spdlog::info("[DX12Adapter] Name: {}", narrow(m_desc.Description));
        spdlog::info("[DX12Adapter] Dedicated VRAM: {} MB", dedicatedMb);
        spdlog::info(
            "[DX12Adapter] VendorId={:#x}, DeviceId={:#x}",
            m_desc.VendorId,
            m_desc.DeviceId);
    }
}
