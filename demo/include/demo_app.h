// demo/include/sandbox_app.h
#include "image_continuum/core/app_base.h"
#include "game_layer.h"


class DemoApp : public ic::AppBase
{
public:
    DemoApp()
        : AppBase(createSpecification())
    {
    }
    void onInit() override;
    void onUpdate(float dt) override;
    void onShutdown() override;

	GameLayer& getGameLayer() { return m_gameLayer; }

private:
    static ic::AppSpecification createSpecification()
    {
        ic::AppSpecification spec;

        spec.appName = "Demo App";
        spec.window.width = 1920;
        spec.window.height = 1080;
        return spec;
    }

	GameLayer m_gameLayer;
};