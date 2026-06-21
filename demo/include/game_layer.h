// game_layer.h
#pragma once
#include "image_continuum/interface/layer.h"

struct ic::Event;

class GameLayer : public ic::Layer 
{
public:
    void onUpdate(float dt);
    void onRender(float alpha);
    void onEvent(ic::Event& e);
};