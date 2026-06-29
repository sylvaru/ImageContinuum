#pragma once

#include <dxgi1_6.h>
#include <wrl/client.h>

namespace ic
{
    class DX12Factory;

    class DX12Adapter
    {
    public:
        void init(const DX12Factory& factory);
        void shutdown();

        IDXGIAdapter4* adapter() const
        {
            return m_adapter.Get();
        }

        const DXGI_ADAPTER_DESC3& desc() const
        {
            return m_desc;
        }

    private:
        bool isAdapterSuitable(IDXGIAdapter4* adapter) const;
        void logAdapterInfo() const;

        Microsoft::WRL::ComPtr<IDXGIAdapter4> m_adapter;
        DXGI_ADAPTER_DESC3 m_desc{};
    };
}
