#include "ic/common/ic_pch.h"
#include "ic/core/app_config.h"

#include <toml++/toml.hpp>
#include <spdlog/spdlog.h>

#include <cctype>
#include <stdexcept>

namespace
{
    bool isDeveloperBuild()
    {
#ifdef IC_DEVELOPER_BUILD
        return true;
#else
        return false;
#endif
    }

    std::filesystem::path resolveProjectPath(const std::filesystem::path& path)
    {
        if (path.is_absolute())
        {
            return path;
        }

        std::filesystem::path candidate = path;
        if (std::filesystem::exists(candidate))
        {
            return candidate;
        }

        std::filesystem::path cursor = std::filesystem::current_path();
        while (!cursor.empty())
        {
            candidate = (cursor / path).lexically_normal();
            if (std::filesystem::exists(candidate))
            {
                return candidate;
            }

            const std::filesystem::path parent = cursor.parent_path();
            if (parent == cursor)
            {
                break;
            }
            cursor = parent;
        }

        return path;
    }

    std::string lower(std::string value)
    {
        for (char& c : value)
        {
            c = static_cast<char>(
                std::tolower(static_cast<unsigned char>(c)));
        }
        return value;
    }

    std::filesystem::path configPathFromArgs(
        const ic::AppConfigLoadDesc& desc,
        int argc,
        char** argv)
    {
        for (int i = 1; i < argc; ++i)
        {
            const std::string_view arg = argv[i] ? argv[i] : "";
            constexpr std::string_view prefix = "--config=";
            if (arg == "--config" && i + 1 < argc)
            {
                return argv[i + 1];
            }

            if (arg.starts_with(prefix))
            {
                return std::string(arg.substr(prefix.size()));
            }
        }

        return desc.defaultConfigPath;
    }

    std::string stringOr(
        const toml::table& table,
        std::string_view key,
        std::string_view fallback)
    {
        return table[key].value_or<std::string>(std::string(fallback));
    }

    uint32_t uintOr(
        const toml::table& table,
        std::string_view key,
        uint32_t fallback)
    {
        if (const std::optional<int64_t> value = table[key].value<int64_t>())
        {
            if (*value < 0)
            {
                throw std::runtime_error(
                    "Config value must be non-negative: " +
                    std::string(key));
            }
            return static_cast<uint32_t>(*value);
        }
        return fallback;
    }

    bool boolOr(
        const toml::table& table,
        std::string_view key,
        bool fallback)
    {
        if (const std::optional<bool> value = table[key].value<bool>())
        {
            return *value;
        }
        return fallback;
    }

    std::filesystem::path pathOr(
        const toml::table& table,
        std::string_view key,
        const std::filesystem::path& fallback)
    {
        if (const std::optional<std::string> value =
            table[key].value<std::string>())
        {
            return *value;
        }
        return fallback;
    }

    float floatOr(
        const toml::table& table,
        std::string_view key,
        float fallback)
    {
        if (const std::optional<double> value = table[key].value<double>())
        {
            return static_cast<float>(*value);
        }
        if (const std::optional<int64_t> value = table[key].value<int64_t>())
        {
            return static_cast<float>(*value);
        }
        return fallback;
    }

    void readEnvironmentSettings(
        const toml::table& table,
        ic::EnvironmentSettings& settings)
    {
        settings.enabled = boolOr(table, "enabled", settings.enabled);
        settings.intensity = floatOr(table, "intensity", settings.intensity);
        settings.skyboxExposure =
            floatOr(table, "skybox_exposure", settings.skyboxExposure);
        settings.pathTraceExposure =
            floatOr(table, "path_trace_exposure", settings.pathTraceExposure);
        settings.tonemapExposure =
            floatOr(table, "tonemap_exposure", settings.tonemapExposure);
        settings.cubemapSize =
            uintOr(table, "cubemap_size", settings.cubemapSize);
    }

    ic::RendererBackendType parseBackend(std::string value)
    {
        value = lower(std::move(value));
        if (value == "vulkan")
        {
            return ic::RendererBackendType::Vulkan;
        }

#ifdef _WIN32
        if (value == "dx12" || value == "directx12" || value == "d3d12")
        {
            return ic::RendererBackendType::DX12;
        }
#endif

        throw std::runtime_error("Unknown renderer backend: " + value);
    }

    ic::WindowMode parseWindowMode(std::string value)
    {
        value = lower(std::move(value));

        if (value == "windowed")
        {
            return ic::WindowMode::Windowed;
        }

        if (value == "maximized")
        {
            return ic::WindowMode::Maximized;
        }

        if (value == "borderless" ||
            value == "borderless_fullscreen" ||
            value == "borderless-fullscreen")
        {
            return ic::WindowMode::BorderlessFullscreen;
        }

        if (value == "fullscreen" ||
            value == "exclusive_fullscreen" ||
            value == "exclusive-fullscreen")
        {
            return ic::WindowMode::Fullscreen;
        }

        throw std::runtime_error("Unknown window mode: " + value);
    }

    ic::RenderPathType parseRenderPath(std::string value)
    {
        value = lower(std::move(value));
        if (value == "forward")
        {
            return ic::RenderPathType::Forward;
        }

        if (value == "deferred")
        {
            throw std::runtime_error(
                "Deferred render path is not implemented yet.");
        }

        if (value == "forward_plus" ||
            value == "forwardplus" ||
            value == "forward-plus" ||
            value == "clusteredforward" ||
            value == "clustered_forward" ||
            value == "clustered-forward")
        {
            return ic::RenderPathType::ClusteredForward;
        }

        if (value == "pathtraced" ||
            value == "path_traced" ||
            value == "path_tracer" ||
            value == "path_trace" ||
            value == "pathtrace" ||
            value == "path-trace" ||
            value == "path-traced" ||
            value == "pathtracing" ||
            value == "path_tracing" ||
            value == "path-tracing")
        {
            return ic::RenderPathType::PathTraced;
        }

        throw std::runtime_error("Unknown render path: " + value);
    }

    ic::GpuCullDebugMode parseGpuCullDebugMode(std::string value)
    {
        value = lower(std::move(value));
        if (value == "off")
        {
            return ic::GpuCullDebugMode::Off;
        }
        if (value == "statistics" || value == "stats")
        {
            return ic::GpuCullDebugMode::Statistics;
        }
        if (value == "classification" ||
            value == "classification_view")
        {
            return ic::GpuCullDebugMode::Classification;
        }
        throw std::runtime_error(
            "Unknown GPU cull debug mode: " + value);
    }
}

namespace ic
{
    RendererBackendType defaultRendererBackend()
    {
#ifdef _WIN32
        return RendererBackendType::DX12;
#else
        return RendererBackendType::Vulkan;
#endif
    }

    AppConfig loadAppConfig(
        const AppConfigLoadDesc& desc,
        int argc,
        char** argv)
    {
        const std::filesystem::path requestedPath =
            configPathFromArgs(desc, argc, argv);
        const std::filesystem::path configPath =
            resolveProjectPath(requestedPath);

        if (!std::filesystem::exists(configPath))
        {
            throw std::runtime_error(
                "App config file not found: " + requestedPath.string() +
                " (resolved: " + configPath.string() + ")");
        }

        toml::table root{};
        try
        {
            root = toml::parse_file(configPath.string());
        }
        catch (const toml::parse_error& e)
        {
            throw std::runtime_error(
                "Failed to parse app config '" + requestedPath.string() +
                "': " + std::string(e.description()));
        }

        AppConfig config{};
        config.app = desc.fallbackApp;
        config.app.rendererSpec.backendType = defaultRendererBackend();
        config.startupScenePath = desc.fallbackStartupScenePath;

        if (const toml::table* app = root["app"].as_table())
        {
            config.app.appName =
                stringOr(*app, "name", config.app.appName);
        }

        if (const toml::table* window = root["window"].as_table())
        {
            config.app.window.width =
                uintOr(*window, "width", config.app.window.width);
            config.app.window.height =
                uintOr(*window, "height", config.app.window.height);
            config.app.window.resizable =
                boolOr(*window, "resizable", config.app.window.resizable);
            config.app.window.maximized =
                boolOr(*window, "maximized", config.app.window.maximized);
            config.app.window.fullscreen =
                boolOr(*window, "fullscreen", config.app.window.fullscreen);

            if (const std::optional<std::string> mode =
                (*window)["mode"].value<std::string>())
            {
                config.app.window.mode = parseWindowMode(*mode);
            }
            else if (config.app.window.fullscreen)
            {
                config.app.window.mode = WindowMode::Fullscreen;
            }
            else if (config.app.window.maximized)
            {
                config.app.window.mode = WindowMode::Maximized;
            }
        }

        if (const toml::table* paths = root["paths"].as_table())
        {
            config.app.resourceRoots.assetRoot =
                pathOr(
                    *paths,
                    "asset_root",
                    config.app.resourceRoots.assetRoot);
            config.app.resourceRoots.modelRoot =
                pathOr(
                    *paths,
                    "model_root",
                    config.app.resourceRoots.modelRoot);
        }

        if (const toml::table* renderer = root["renderer"].as_table())
        {
            if (const std::optional<std::string> backend =
                (*renderer)["backend"].value<std::string>())
            {
                if (isDeveloperBuild())
                {
                    config.app.rendererSpec.backendType =
                        parseBackend(*backend);
                }
                else
                {
                    spdlog::warn(
                        "[AppConfig] Ignoring renderer.backend outside a developer build.");
                }
            }

            if (const std::optional<std::string> path =
                (*renderer)["path"].value<std::string>())
            {
                config.app.rendererSpec.pathType =
                    parseRenderPath(*path);
            }
            config.app.rendererSpec.enableValidation =
                boolOr(
                    *renderer,
                    "validation",
                    config.app.rendererSpec.enableValidation);
            config.app.rendererSpec.useDebugGui =
                boolOr(
                    *renderer,
                    "debug_gui",
                    config.app.rendererSpec.useDebugGui);
            config.app.rendererSpec.settings.vsync =
                boolOr(
                    *renderer,
                    "vsync",
                    config.app.rendererSpec.settings.vsync);
            config.app.rendererSpec.settings.targetFps =
                floatOr(
                    *renderer,
                    "target_fps",
                    config.app.rendererSpec.settings.targetFps);
            config.app.rendererSpec.settings.gpuOcclusion =
                boolOr(
                    *renderer,
                    "gpu_occlusion",
                    config.app.rendererSpec.settings.gpuOcclusion);
            if (const std::optional<std::string> debugMode =
                (*renderer)["gpu_cull_debug"].value<std::string>())
            {
                config.app.rendererSpec.settings.gpuCullDebugMode =
                    parseGpuCullDebugMode(*debugMode);
            }
            if (const toml::table* environment =
                    renderer->get_as<toml::table>("environment"))
            {
                readEnvironmentSettings(
                    *environment,
                    config.app.rendererSpec.settings.environment);
            }
            config.app.rendererSpec.framesInFlight =
                uintOr(
                    *renderer,
                    "frames_in_flight",
                    config.app.rendererSpec.framesInFlight);

            if (const std::optional<std::string> pipelineLibrary =
                (*renderer)["pipeline_library"].value<std::string>())
            {
                config.app.rendererSpec.pipelineLibraryPath =
                    *pipelineLibrary;
            }
        }

        if (const toml::table* scene = root["scene"].as_table())
        {
            config.startupScenePath =
                stringOr(*scene, "file", config.startupScenePath.string());
        }

        spdlog::info("[AppConfig] Loaded {}", configPath.string());
        spdlog::info(
            "[AppConfig] Pipeline library: {}",
            config.app.rendererSpec.pipelineLibraryPath.string());
        spdlog::info(
            "[AppConfig] Startup scene: {}",
            config.startupScenePath.string());
        spdlog::info(
            "[AppConfig] Asset root: {}",
            config.app.resourceRoots.assetRoot.string());
        spdlog::info(
            "[AppConfig] Model root: {}",
            config.app.resourceRoots.modelRoot.string());

        return config;
    }
}
