// demo/src/demo_app.cpp
#include "common/demo_pch.h"
#include "demo_app.h"

void DemoApp::onInit(ic::AppBase<DemoApp>& appInstance)
{
	spdlog::info("[ Demo App ] onInit..");
	GameLayer layer;
	appInstance.pushLayerToStack(layer);
}

void DemoApp::onRender(ic::AppBase<DemoApp>& appInstance)
{
	(void)appInstance;
	spdlog::info("[ Demo App ] onRender..");
}

void DemoApp::onUpdate(ic::AppBase<DemoApp>& appInstance)
{
	(void)appInstance;
	spdlog::info("[ Demo App ] onUpdate..");
}

void DemoApp::onShutdown(ic::AppBase<DemoApp>& appInstance)
{
	(void)appInstance;
	spdlog::info("[ Demo App ] onShutdown..");
}