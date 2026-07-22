#include "common/tests_pch.h"
#include "ray_tracing_scene_tests.h"
#include "ic/renderer/ray_tracing/ray_tracing_scene.h"
#include <spdlog/spdlog.h>

using namespace ic;
namespace { int failures = 0; void check(bool v, const char* m) { if (!v) { ++failures; spdlog::error("[RayTracingSceneTests] FAIL: {}", m); } } }

int runRayTracingSceneTests()
{
    failures = 0;
    check(classifyRayTracingSceneUpdate(false, 0, 1, 0, 1, 0, 1) == RayTracingSceneUpdateKind::GeometryRebuild, "initial build");
    check(classifyRayTracingSceneUpdate(true, 1, 1, 2, 2, 3, 3) == RayTracingSceneUpdateKind::None, "static reuse");
    check(classifyRayTracingSceneUpdate(true, 1, 1, 2, 2, 3, 4) == RayTracingSceneUpdateKind::TlasRefit, "transform/generation refit");
    check(classifyRayTracingSceneUpdate(true, 1, 1, 2, 3, 3, 4) == RayTracingSceneUpdateKind::TlasRebuild, "instance topology rebuild");
    check(classifyRayTracingSceneUpdate(true, 1, 2, 2, 2, 3, 3) == RayTracingSceneUpdateKind::GeometryRebuild, "geometry rebuild");
    MaterialAsset opaque{};
    check(rayTracingGeometryFlags(&opaque) == RayTracingGeometryOpaque, "opaque metadata");
    MaterialAsset alpha{}; alpha.alphaMask = true;
    check((rayTracingGeometryFlags(&alpha) & RayTracingGeometryAlphaTested) != 0 && (rayTracingGeometryFlags(&alpha) & RayTracingGeometryOpaque) == 0, "alpha metadata");
    MaterialAsset twoSided{}; twoSided.doubleSided = true;
    check((rayTracingGeometryFlags(&twoSided) & RayTracingGeometryDoubleSided) != 0, "double-sided metadata");
    check(sizeof(GpuRayTracingGeometryRecord) == 48 &&
          offsetof(GpuRayTracingGeometryRecord, mapping) == 16 &&
          offsetof(GpuRayTracingGeometryRecord, source) == 32,
        "GPU geometry mapping matches common HLSL layout");
    check(sizeof(GpuRayTracingInstanceRecord) == 160 &&
          offsetof(GpuRayTracingInstanceRecord, identity) == 128 &&
          offsetof(GpuRayTracingInstanceRecord, state) == 144,
        "GPU instance generation/transform mapping matches common HLSL layout");
    check(sizeof(AssetVertex) == 72 && offsetof(AssetVertex, normal) == 12 &&
          offsetof(AssetVertex, tangent) == 24 && offsetof(AssetVertex, uv0) == 40,
        "ray-query vertex reconstruction matches resident raster vertex layout");
    if (failures == 0) spdlog::warn("[RayTracingSceneTests] PASSED");
    return failures;
}
