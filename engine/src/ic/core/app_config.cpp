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

    ic::GlobalIlluminationDebugView parseGlobalIlluminationDebugView(
        std::string value)
    {
        value = lower(std::move(value));
        if (value == "none" || value == "off")
            return ic::GlobalIlluminationDebugView::None;
        if (value == "irradiance" || value == "irradiance_only")
            return ic::GlobalIlluminationDebugView::IrradianceOnly;
        if (value == "contribution" || value == "contribution_only")
            return ic::GlobalIlluminationDebugView::ContributionOnly;
        if (value == "coverage" || value == "coverage_confidence")
            return ic::GlobalIlluminationDebugView::CoverageConfidence;
        if (value == "comparison" || value == "enabled_disabled")
            return ic::GlobalIlluminationDebugView::EnabledDisabledComparison;
        if (value == "diagnostic_intensity" || value == "intensity")
            return ic::GlobalIlluminationDebugView::DiagnosticIntensity;
        if (value == "traced_rays")
            return ic::GlobalIlluminationDebugView::TracedRays;
        if (value == "hit_distance")
            return ic::GlobalIlluminationDebugView::HitDistance;
        if (value == "reconstructed_normals")
            return ic::GlobalIlluminationDebugView::ReconstructedNormals;
        if (value == "reconstructed_materials")
            return ic::GlobalIlluminationDebugView::ReconstructedMaterials;
        if (value == "clipmap_probe_identity" || value == "selected_surfel" ||
            value == "selected_surfel_id")
            return ic::GlobalIlluminationDebugView::ClipmapLevelProbeIdentity;
        if (value == "probe_classification_relocation" ||
            value == "surfel_position" || value == "surfel_world_position")
            return ic::GlobalIlluminationDebugView::ProbeClassificationRelocation;
        if (value == "probe_depth_visibility_moments" ||
            value == "surfel_normal" || value == "surfel_world_normal")
            return ic::GlobalIlluminationDebugView::ProbeDepthVisibilityMoments;
        if (value == "probe_radiance_sh" || value == "raw_sh" ||
            value == "raw_sh_coefficients")
            return ic::GlobalIlluminationDebugView::ProbeRadianceSh;
        if (value == "current_previous" ||
            value == "current_previous_irradiance")
            return ic::GlobalIlluminationDebugView::CurrentPreviousIrradiance;
        if (value == "temporal_rejection" ||
            value == "temporal_rejection_reason")
            return ic::GlobalIlluminationDebugView::TemporalRejectionReason;
        if (value == "probe_age_update_state" || value == "surfel_age" ||
            value == "allocation_update_age")
            return ic::GlobalIlluminationDebugView::ProbeAgeUpdateState;
        if (value == "probe_confidence_variance" || value == "variance" ||
            value == "variance_confidence")
            return ic::GlobalIlluminationDebugView::ProbeConfidenceVariance;
        if (value == "raw_gathered_half_resolution" || value == "raw_half" ||
            value == "raw_half_resolution")
            return ic::GlobalIlluminationDebugView::RawGatheredHalfResolution;
        if (value == "probe_update_workload" || value == "raw_traced_probe_radiance" || value == "raw_traced_indirect" ||
            value == "raw_traced_indirect_radiance")
            return ic::GlobalIlluminationDebugView::ProbeUpdateWorkload;
        if (value == "cached_probe_irradiance" || value == "cached_surfel" ||
            value == "cached_surfel_radiance")
            return ic::GlobalIlluminationDebugView::CachedProbeIrradiance;
        if (value == "gathered_irradiance_coverage" || value == "gathered" ||
            value == "gathered_irradiance")
            return ic::GlobalIlluminationDebugView::GatheredIrradianceCoverage;
        if (value == "temporal_filtered_irradiance" || value == "temporal" ||
            value == "temporal_result")
            return ic::GlobalIlluminationDebugView::TemporalFilteredIrradiance;
        if (value == "direct" || value == "direct_only")
            return ic::GlobalIlluminationDebugView::DirectOnly;
        if (value == "temporal_weight" || value == "history_weight" ||
            value == "temporal_history_weight")
            return ic::GlobalIlluminationDebugView::TemporalHistoryWeight;
        throw std::runtime_error("Unknown global illumination debug view: " + value);
    }

    ic::GlobalIlluminationQuality parseGlobalIlluminationQuality(
        std::string value)
    {
        value = lower(std::move(value));
        if (value == "off") return ic::GlobalIlluminationQuality::Off;
        if (value == "low") return ic::GlobalIlluminationQuality::Low;
        if (value == "medium") return ic::GlobalIlluminationQuality::Medium;
        if (value == "high") return ic::GlobalIlluminationQuality::High;
        if (value == "ultra") return ic::GlobalIlluminationQuality::Ultra;
        if (value == "custom") return ic::GlobalIlluminationQuality::Custom;
        throw std::runtime_error("Unknown global illumination quality: " + value);
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
            config.app.rendererSpec.settings.rayTracing =
                boolOr(*renderer, "ray_tracing",
                    config.app.rendererSpec.settings.rayTracing);
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
            if (const toml::table* gi =
                    renderer->get_as<toml::table>("global_illumination"))
            {
                auto& settings =
                    config.app.rendererSpec.settings.globalIllumination;
                if (const std::optional<std::string> quality =
                    (*gi)["quality"].value<std::string>())
                {
                    settings = globalIlluminationPreset(
                        parseGlobalIlluminationQuality(*quality), settings);
                }
                settings.enabled = boolOr(*gi, "enabled", settings.enabled);
                settings.asyncCompute =
                    boolOr(*gi, "async_compute", settings.asyncCompute);
                settings.diagnosticsReadback = boolOr(
                    *gi, "diagnostics_readback", settings.diagnosticsReadback);
                settings.evaluationDivisor = uintOr(
                    *gi, "evaluation_divisor", settings.evaluationDivisor);
                settings.evaluationDivisor = uintOr(
                    *gi, "temporal_reconstruction_divisor",
                    settings.evaluationDivisor);
                settings.gpuTimeTargetMilliseconds = std::clamp(floatOr(
                    *gi, "gpu_time_target_ms",
                    settings.gpuTimeTargetMilliseconds), 0.0f, 1000.0f);
                const uint64_t memoryLimitMiB = uintOr(
                    *gi, "memory_limit_mb", static_cast<uint32_t>(
                        settings.memoryLimitBytes / (1024ull * 1024ull)));
                settings.memoryLimitBytes = memoryLimitMiB * 1024ull * 1024ull;
                settings.surfelCellSize = std::max(floatOr(
                    *gi, "surfel_cell_size", settings.surfelCellSize),
                    1.0e-3f);
                settings.surfelDetail = boolOr(
                    *gi, "surfel_detail", settings.surfelDetail);
                settings.diagnosticIntensity = std::clamp(floatOr(
                    *gi, "diagnostic_intensity", settings.diagnosticIntensity),
                    0.0f, 16.0f);
                settings.debugExposure = std::clamp(floatOr(
                    *gi, "debug_exposure", settings.debugExposure),
                    0.03125f, 32.0f);
                settings.freezeAfterFrames = uintOr(
                    *gi, "freeze_after_frames", settings.freezeAfterFrames);
                settings.multiBounce = boolOr(
                    *gi, "multi_bounce", settings.multiBounce);
                settings.probeFallback = boolOr(
                    *gi, "probe_fallback", settings.probeFallback);
                if (const std::optional<std::string> debugView =
                    (*gi)["debug_view"].value<std::string>())
                {
                    settings.debugView =
                        parseGlobalIlluminationDebugView(*debugView);
                }
                settings.limits.maxSurfels = uintOr(
                    *gi, "max_surfels", settings.limits.maxSurfels);
                settings.limits.hashBucketCount = uintOr(
                    *gi, "hash_buckets", settings.limits.hashBucketCount);
                settings.limits.maxSurfelUpdates = uintOr(
                    *gi, "max_surfel_updates",
                    settings.limits.maxSurfelUpdates);
                settings.limits.probeClipmapCount = uintOr(
                    *gi, "probe_clipmaps",
                    settings.limits.probeClipmapCount);
                settings.limits.probeResolution = uintOr(
                    *gi, "probe_resolution",
                    settings.limits.probeResolution);
                settings.limits.maxProbeUpdates = uintOr(
                    *gi, "max_probe_updates",
                    settings.limits.maxProbeUpdates);
                settings.limits.rayBudget = uintOr(
                    *gi, "ray_budget", settings.limits.rayBudget);
                if (const auto visibilityRays =
                    (*gi)["visibility_rays_per_probe"].value<uint32_t>())
                {
                    const uint64_t budget = static_cast<uint64_t>(
                        settings.limits.maxProbeUpdates) *
                        std::clamp(*visibilityRays, 1u, 32u);
                    settings.limits.rayBudget = static_cast<uint32_t>(
                        std::min<uint64_t>(budget,
                            std::numeric_limits<uint32_t>::max()));
                }
                if (settings.quality != GlobalIlluminationQuality::Off &&
                    settings.quality != GlobalIlluminationQuality::Custom)
                {
                    const auto preset = globalIlluminationPreset(
                        settings.quality, settings);
                    const auto& a = settings.limits;
                    const auto& b = preset.limits;
                    if (a.maxSurfels != b.maxSurfels ||
                        a.hashBucketCount != b.hashBucketCount ||
                        a.maxSurfelUpdates != b.maxSurfelUpdates ||
                        a.probeClipmapCount != b.probeClipmapCount ||
                        a.probeResolution != b.probeResolution ||
                        a.maxProbeUpdates != b.maxProbeUpdates ||
                        a.rayBudget != b.rayBudget ||
                        settings.evaluationDivisor != preset.evaluationDivisor ||
                        settings.surfelCellSize != preset.surfelCellSize ||
                        settings.surfelDetail != preset.surfelDetail ||
                        settings.multiBounce != preset.multiBounce ||
                        settings.probeFallback != preset.probeFallback)
                    {
                        settings.quality = GlobalIlluminationQuality::Custom;
                    }
                }
                if (!settings.enabled)
                    settings.quality = GlobalIlluminationQuality::Off;
                else if (settings.quality == GlobalIlluminationQuality::Off)
                    settings.enabled = false;
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
