// demo/src/demo_app.cpp
#include "common/demo_pch.h"
#include "demo_app.h"
#include <spdlog/spdlog.h>

void DemoApp::onInit()
{
	spdlog::info("[ Demo App ] onInit..");
	m_gameLayer = &pushLayer<GameLayer>(m_config.startupScenePath);
}

void DemoApp::onUpdate([[maybe_unused]] float dt)
{

}

void DemoApp::onShutdown()
{
	spdlog::info("[ Demo App ] onShutdown..");
}
