// demo/src/game_layer.cpp
#pragma once
#include "game_layer.h"
#include "image_continuum/interface/events.h"


void GameLayer::onUpdate(float dt) { 
	(void)dt;/* Perform math here */ 
}
void GameLayer::onRender(float alpha) { 
	(void)alpha;/* Submit flat data buffers */ 
}
void GameLayer::onEvent(ic::Event& e) {
	(void)e;/* */
}