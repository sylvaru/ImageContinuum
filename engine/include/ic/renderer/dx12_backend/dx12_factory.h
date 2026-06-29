#pragma once

#include <dxgi1_6.h>
#include <wrl/client.h>

namespace ic
{
    class DX12Factory
    {
    public:
        void init(bool enableValidation);
        void shutdown();

        IDXGIFactory6* factory() const
        {
            return m_factory.Get();
        }

        bool allowTearing() const
        {
            return m_allowTearing;
        }

        bool validationEnabled() const
        {
            return m_validationEnabled;
        }

    private:
        Microsoft::WRL::ComPtr<IDXGIFactory6> m_factory;
        bool m_validationEnabled = false;
        bool m_allowTearing = false;
    };
}
