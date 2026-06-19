// game_layer.h
#pragma once
#include "image_continuum/interface/layer.h"


class GameLayer : public ic::Layer<GameLayer> {
public:
    void onUpdate(float dt) { (void)dt;/* Perform math here */ }
    void onRender(float alpha) { (void)alpha;/* Submit flat data buffers */ }
};