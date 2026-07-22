#include "common/tests_pch.h"
#include "job_system_stress_tests.h"
#include "frame_graph_resource_tests.h"
#include "pass_dispatch_tests.h"
#include "global_illumination_tests.h"
#include "ray_tracing_scene_tests.h"

#include <spdlog/spdlog.h>

int main()
{
    // Quiet the per-init/shutdown info spam so the per-test progress markers on
    // stderr stay attributable; warnings and job-exception errors still surface.
    spdlog::set_level(spdlog::level::warn);

    int failures = 0;
    failures += runJobSystemStressTests();
    failures += runFrameGraphResourceTests();
    failures += runPassDispatchTests();
    failures += runGlobalIlluminationTests();
    failures += runRayTracingSceneTests();
    return failures;
}
