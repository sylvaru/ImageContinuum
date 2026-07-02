// game_layer.h
#pragma once
#include "ic/interface/layer.h"
#include "ic/scene/scene_types.h"

#include <filesystem>

using namespace ic;

struct Event;
struct FrameContext;

class GameLayer : public Layer 
{
public:
    explicit GameLayer(std::filesystem::path scenePath);

    void onAttach(AppServices& services) override;
    void onUpdate(ic::FrameContext& ctx) override;
    void onRender(float alpha) override;

private:
    std::filesystem::path m_scenePath;
    SceneHandle m_scene;
};
