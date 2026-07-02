#pragma once

#include "ic/core/app_base.h"

#include <filesystem>

namespace ic
{
    struct AppConfig
    {
        AppSpecification app;
        std::filesystem::path startupScenePath;
    };

    struct AppConfigLoadDesc
    {
        std::filesystem::path defaultConfigPath;
        AppSpecification fallbackApp;
        std::filesystem::path fallbackStartupScenePath;
    };

    RendererBackendType defaultRendererBackend();

    AppConfig loadAppConfig(
        const AppConfigLoadDesc& desc,
        int argc,
        char** argv);
}
