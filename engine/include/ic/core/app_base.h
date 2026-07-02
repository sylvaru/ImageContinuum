// ic/core/app_base.h
#pragma once
#include "ic/common/ic_common.h"
#include "ic/common/ic_fwd.h"
#include "ic/renderer/renderer_specification.h"
#include "ic/interface/window.h"

#include "layer_stack.h"
#include "clock.h"
#include "frame_context.h"


namespace ic 
{

    struct AppResourceRoots
    {
        std::filesystem::path assetRoot = "assets";
        std::filesystem::path modelRoot;
    };

    struct AppSpecification
    {
        std::string appName = "Demo";

        WindowSpecification window;

        RendererSpecification rendererSpec;
        AppResourceRoots resourceRoots;

        static constexpr size_t kMaxFramesInFlight = 2;
    };

    struct AppServices
    {
        Input* input = nullptr;
        Window* window = nullptr;
        Renderer* renderer = nullptr;

        JobSystem* jobSystem = nullptr;
        AssetManager* assetManager = nullptr;
        SceneManager* sceneManager = nullptr;
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
            T& layer = m_layerStack.emplaceLayer<T>(
                std::forward<Args>(args)...);

            layer.onAttach(m_services);
            return layer;
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
        struct AppRuntime;
        struct PlatformState;

        void initAppBase(int argc, char** argv);
        void initRenderer();

        void shutdown();

        void createJobSystem();
        void createPlatform();
        void createWindow();
        void createInput();
        void createFrameArenas();
        void createRenderer();
        void createAssetManager();
        void createSceneManager();

        void buildServices();
        void bindEventSink();

        void sleep(const auto& frameStart);

        void handleInputEvent(Event& e);
        void handleWindowEvent(Event& e);
        void handleRenderEvent(Event& e);

        Scope<AppRuntime> m_runtime;
        AppSpecification m_spec{};
        Clock m_clock;
        
        LayerStack m_layerStack{};
        AppServices m_services{};
        FrameContext m_frame{};
        
        static constexpr float kTargetFPS = 144.0f;
        static constexpr float kTargetFrameTime = 1.0f / kTargetFPS;
    };
}
