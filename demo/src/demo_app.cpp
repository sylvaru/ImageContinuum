// demo/src/demo_app.cpp
#include "common/demo_pch.h"
#include "demo_app.h"
#include <spdlog/spdlog.h>

void DemoApp::onInit()
{
	spdlog::info("[ Demo App ] onInit..");
	pushLayer<GameLayer>();
}

void DemoApp::onUpdate(float dt)
{
    (void)dt;
}

void DemoApp::onShutdown()
{
	spdlog::info("[ Demo App ] onShutdown..");
}

ic::AppSpecification DemoApp::createSpecification()
{
    ic::AppSpecification spec;

    spec.appName = "Demo App";
    spec.window.width = 1920;
    spec.window.height = 1080;
    return spec;
}