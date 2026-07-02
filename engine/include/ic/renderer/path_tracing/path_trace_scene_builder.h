#pragma once

#include "ic/renderer/path_tracing/path_tracer_types.h"

namespace ic
{
    class AssetManager;
    struct SceneRenderView;

    PathTraceSceneData buildPathTraceSceneData(
        const SceneRenderView& scene,
        const AssetManager& assets);
}
