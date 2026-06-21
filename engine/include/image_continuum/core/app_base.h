// image_continuum/core/app_base.h
#pragma once
#include "layer_stack.h"
#include "image_continuum/interface/renderer_backend.h"
#include "image_continuum/interface/window.h"
#include "event_frame_buffer.h"
#include "frame_executor.h"

namespace ic 
{
    struct Input;
    struct Event;

    struct AppServices
    {
        //Renderer* renderer;
        //AssetManager* assetManager;
        //SceneManager* sceneManager;
        Input* input;
        Window* window;
    };


    struct AppSpecification
    {
        std::string appName = "Demo";

        WindowSpecification window;

        RendererSpecification renderer;
    };

    class AppBase
    {
    public:
        AppBase(AppSpecification apec);

        int run(int argc, char** argv);

    protected:

        virtual ~AppBase();
        virtual void onInit() {}
        virtual void onShutdown() {}

        void onEvent(Event& e)
        {
            m_layerStack.dispatchEvent(e);
        }

        template<typename T, typename... Args>
        T& pushLayer(Args&&... args)
        {
            return m_layerStack.emplaceLayer<T>(
                std::forward<Args>(args)...);
        }

    public:
        AppServices& services()
        {
            return m_services;
        }

        LayerStack& layerStack()
        {
            return m_layerStack;
        }
        virtual void onUpdate(float dt) { (void)dt; }

        void dispatchEvent(EventChannel channel, Event& e);

    private:
    private:
        void initializeAppBase();
        void shutdown();

        void createPlatform();
        void createWindow();
        void createInput();
        void bindEvents();
        void buildServices();

        void handleInputEvent(Event& e);
        void handleWindowEvent(Event& e);
        void handleRenderEvent(Event&);

        AppSpecification m_spec{};
        struct PlatformState;
        std::unique_ptr<PlatformState> m_platform;

        std::unique_ptr<Window> m_window;
        std::unique_ptr<Input> m_input;
        
        //std::unique_ptr<RenderBackend> m_renderer;

        LayerStack m_layerStack{};
        AppServices m_services{};
        EventFrameBuffer m_eventFrameBuffer{};
        FrameContext m_frame{};
        FrameExecutor m_executor{ *this };
    };
}