// game_layer.h
#pragma once
#include "ic/interface/layer.h"
#include "ic/scene/scene_types.h"

#include <filesystem>
#include <glm/glm.hpp>

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
    bool m_cameraSweepEnabled = false;
    bool m_cameraSweepInitialized = false;
    glm::vec3 m_cameraSweepOrigin = glm::vec3(0.0f);
    bool m_dynamicSweepEnabled = false;
    bool m_dynamicSweepInitialized = false;
    glm::vec3 m_dynamicSweepOrigin = glm::vec3(0.0f);
};
