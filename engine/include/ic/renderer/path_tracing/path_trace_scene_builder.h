#pragma once

#include <cstdint>
#include <functional>

#include "ic/core/asset_manager.h"
#include "ic/renderer/path_tracing/path_tracer_types.h"

namespace ic
{
    struct SceneRenderView;

    struct PathTraceMaterialTextureIndices
    {
        uint32_t textureIndex = UINT32_MAX;
        uint32_t samplerIndex = UINT32_MAX;
    };

    using PathTraceMaterialTextureResolver =
        std::function<PathTraceMaterialTextureIndices(
            AssetHandle,
            const MaterialTextureSlot&)>;

    PathTraceSceneData buildPathTraceSceneData(
        const SceneRenderView& scene,
        const AssetManager& assets,
        const PathTraceMaterialTextureResolver& resolveTexture = {});
}
