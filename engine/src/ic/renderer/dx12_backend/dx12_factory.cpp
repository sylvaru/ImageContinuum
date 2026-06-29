#include "ic/renderer/dx12_backend/dx12_factory.h"

#include <d3d12.h>
#include <dxgidebug.h>
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
}

namespace ic
{
    void DX12Factory::init(bool enableValidation)
    {
        spdlog::info("[DX12Factory] Initializing...");

        UINT factoryFlags = 0;

        if (enableValidation)
        {
            Microsoft::WRL::ComPtr<ID3D12Debug> debug;
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))))
            {
                debug->EnableDebugLayer();
                factoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
                m_validationEnabled = true;
                spdlog::info("[DX12Factory] D3D12 debug layer enabled.");
            }
            else
            {
                spdlog::warn("[DX12Factory] D3D12 debug layer requested but unavailable.");
            }
        }

        throwIfFailed(
            CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(&m_factory)),
            "Failed to create DXGI factory.");

        Microsoft::WRL::ComPtr<IDXGIFactory5> factory5;
        if (SUCCEEDED(m_factory.As(&factory5)))
        {
            BOOL allowTearing = FALSE;
            if (SUCCEEDED(factory5->CheckFeatureSupport(
                    DXGI_FEATURE_PRESENT_ALLOW_TEARING,
                    &allowTearing,
                    sizeof(allowTearing))))
            {
                m_allowTearing = allowTearing == TRUE;
            }
        }

        spdlog::info(
            "[DX12Factory] Initialized (validation={}, tearing={})",
            m_validationEnabled,
            m_allowTearing);
    }

    void DX12Factory::shutdown()
    {
        m_factory.Reset();
        m_validationEnabled = false;
        m_allowTearing = false;
        spdlog::info("[DX12Factory] Shutdown");
    }
}
