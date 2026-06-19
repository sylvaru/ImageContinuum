// engine/include/image_continuum/core/app_base.h
#pragma once
#include "layer_stack.h"

#include <cstdint>
#include <utility>

namespace ic 
{

    struct AppServices
    {
        //Renderer* renderer
        //AssetManager* assetManager;
        //SceneManager* sceneManager;
    };

    struct AppSpecification
    {
        uint32_t maxScratchMemory;
        bool enableImGui;
    };

    template <typename Derived>
    class AppBase 
    {
    protected:
        Derived& self() noexcept;

    public:
        AppBase() = default;
        ~AppBase() = default;

        AppBase(const AppBase&) = delete;
        AppBase& operator=(const AppBase&) = delete;

        int run(int argc, char** argv) 
        {
            (void)argc;
            (void)argv;

            auto& app = self();

            bool isRunning = true;
            while (isRunning) 
            {
                app.onInit(*this);

                m_stack.updateAll(0.016f);

                app.onUpdate(*this);

                m_stack.renderAll(1.0f);

                isRunning = false;
            }

            app.onShutdown(*this);
            return 0;
        }


        template <typename T>
        void pushLayerToStack(T layer) {
            m_stack.pushLayer(std::move(layer));
        }

        LayerStack& getStack() { return m_stack; }
    private:
        LayerStack m_stack;
    };

    template <typename Derived>
    inline Derived& AppBase<Derived>::self() noexcept
    {
        return reinterpret_cast<Derived&>(*this);
    }
}