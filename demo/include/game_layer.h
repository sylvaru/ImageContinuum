// game_layer.h
#pragma once
#include "ic/interface/layer.h"

struct ic::Event;
struct ic::FrameContext;

class GameLayer : public ic::Layer 
{
public:
    void onUpdate(ic::FrameContext& ctx) override;
    void onRender(float alpha) override;
    //void onEvent(ic::Event& e);
};