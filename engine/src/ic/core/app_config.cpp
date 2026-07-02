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

        if (value == "pathtraced" ||
            value == "path_traced" ||
            value == "path-traced" ||
            value == "pathtracing" ||
            value == "path_tracing" ||
            value == "path-tracing")
        {
            return ic::RenderPathType::PathTraced;
        }

        throw std::runtime_error("Unknown render path: " + value);
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
