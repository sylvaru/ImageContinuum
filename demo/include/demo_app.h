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
    static ic::AppSpecification createSpecification();

	GameLayer m_gameLayer;
};