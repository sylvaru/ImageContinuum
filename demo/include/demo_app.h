// demo/include/sandbox_app.h
#include "ic/core/app_base.h"
#include "game_layer.h"
#include "ic/renderer/renderer_specification.h"

using namespace ic;

class DemoApp : public AppBase
{
public:
    DemoApp()
        : AppBase(createSpecification())
    {
    }
    void onInit() override;
    void onUpdate(float dt) override;
    void onShutdown() override;

	GameLayer* getGameLayer() { return m_gameLayer; }

private:
    static AppSpecification createSpecification()
    {
        AppSpecification spec;

        spec.appName = "Demo App";
#ifdef _WIN32
        spec.rendererSpec.backendType = RendererBackendType::DX12;
#else
        spec.rendererSpec.backendType = RendererBackendType::Vulkan;
#endif
        spec.rendererSpec.pathType = RenderPathType::Forward;
        spec.window.width = 1920;
        spec.window.height = 1080;
        return spec;
    }

	GameLayer* m_gameLayer = nullptr;
};
