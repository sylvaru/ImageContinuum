// demo/include/sandbox_app.h
#include "image_continuum/core/app_base.h"
#include "game_layer.h"


class DemoApp : public ic::AppBase<DemoApp>
{
public:
	void onInit(ic::AppBase<DemoApp>& appInstance);
	void onRender(ic::AppBase<DemoApp>& appInstance);
	void onUpdate(ic::AppBase<DemoApp>& appInstance);
	void onShutdown(ic::AppBase<DemoApp>& appInstance);

	GameLayer& getGameLayer() { return m_gameLayer; }

private:
	GameLayer m_gameLayer;
};