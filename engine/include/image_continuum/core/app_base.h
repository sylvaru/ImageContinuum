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
    public:
        AppBase() = default;
        ~AppBase() = default;

        AppBase(const AppBase&) = delete;
        AppBase& operator=(const AppBase&) = delete;

        int run(int argc, char** argv) 
        {
            (void)argc;
            (void)argv;

            Derived::onInit(*this);

            bool isRunning = true;
            while (isRunning) 
            {
                m_stack.updateAndRenderAll(0.016f, 1.0f);

                isRunning = false;
            }

            Derived::onShutdown();
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
}