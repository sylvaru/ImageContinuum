// demo/src/main.cpp
#include "image_continuum/core/app_base.h"
#include "game_layer.h"
#include <iostream> // spdlog will replace this


class SandboxApp : public ic::AppBase<SandboxApp> 
{
public:
	static void onInit(ic::AppBase<SandboxApp>& appInstance) {
        GameLayer layer;
		appInstance.pushLayerToStack(layer);
	}
	void onRender() {}
	void onUpdate() {}

	static void onShutdown() {
		std::cout << "Sandbox Shutdown." << "\n";
	}
	GameLayer& getGameLayer() { return m_gameLayer; }

private:
	GameLayer m_gameLayer;
};

int main(int argc, char** argv) {

	SandboxApp app;

    return app.run(argc, argv);
}