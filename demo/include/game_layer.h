// game_layer.h
#pragma once
#include "ic/interface/layer.h"
#include "ic/scene/scene_types.h"

using namespace ic;

struct Event;
struct FrameContext;

class GameLayer : public Layer 
{
public:
    void onAttach(AppServices& services) override;
    void onUpdate(ic::FrameContext& ctx) override;
    void onRender(float alpha) override;

private:
    SceneHandle m_scene;
};
