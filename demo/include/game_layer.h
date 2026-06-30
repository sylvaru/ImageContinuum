// game_layer.h
#pragma once
#include "ic/interface/layer.h"
#include "ic/scene/scene_types.h"

struct ic::Event;
struct ic::FrameContext;

class GameLayer : public ic::Layer 
{
public:
    void onAttach(ic::AppServices& services) override;
    void onUpdate(ic::FrameContext& ctx) override;
    void onRender(float alpha) override;

private:
    ic::SceneHandle m_scene;
};
