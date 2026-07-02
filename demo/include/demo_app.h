// demo/include/sandbox_app.h
#include "ic/core/app_base.h"
#include "ic/core/app_config.h"
#include "game_layer.h"

#include <cstdlib>
#include <utility>

using namespace ic;

class DemoApp : public AppBase
{
public:
    DemoApp(int argc, char** argv)
        : DemoApp(loadAppConfig(createConfigLoadDesc(), argc, argv))
    {
    }

    void onInit() override;
    void onUpdate(float dt) override;
    void onShutdown() override;

	GameLayer* getGameLayer() { return m_gameLayer; }

private:
    static AppConfigLoadDesc createConfigLoadDesc()
    {
        AppConfigLoadDesc desc{};
        desc.defaultConfigPath =
            "demo/res/configs/path_traced.toml";
        desc.fallbackApp = createFallbackSpecification();
        desc.fallbackStartupScenePath =
            "demo/res/scene_files/cornell_scene.toml";
        return desc;
    }

    static AppSpecification createFallbackSpecification()
    {
        AppSpecification spec{};
        spec.appName = "ImageContinuum Example App";
        spec.rendererSpec.backendType = defaultRendererBackend();
        spec.rendererSpec.pathType = RenderPathType::PathTraced;
        spec.rendererSpec.useDebugGui = true;
        spec.rendererSpec.enableValidation = true;
        spec.rendererSpec.framesInFlight = 2;
        spec.rendererSpec.pipelineLibraryPath =
            "demo/res/pipelines/path_tracing.toml";
        spec.resourceRoots.assetRoot = "demo/res";
        spec.resourceRoots.modelRoot = "demo/res/models";
        spec.window.width = 1920;
        spec.window.height = 1080;
        return spec;
    }

    explicit DemoApp(AppConfig config)
        : AppBase(config.app)
        , m_config(std::move(config))
    {
    }

    AppConfig m_config;
	GameLayer* m_gameLayer = nullptr;
};
