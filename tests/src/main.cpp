#include "common/tests_pch.h"
#include "job_system_stress_tests.h"

#include <spdlog/spdlog.h>

int main()
{
    // Quiet the per-init/shutdown info spam so the per-test progress markers on
    // stderr stay attributable; warnings and job-exception errors still surface.
    spdlog::set_level(spdlog::level::warn);
    return runJobSystemStressTests();
}
