#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string_view>

#include <glm/glm.hpp>

#include "ic/renderer/frame_graph/frame_graph_types.h"
#include "ic/renderer/ray_tracing/ray_tracing_scene.h"
#include "ic/renderer/renderer_path/renderer_path.h"

namespace ic
{
    class FrameGraphBuilder;

    enum class GlobalIlluminationQuality : uint8_t
    {
        Off,
        Low,
        Medium,
        High,
        Ultra,
        Custom,
        Count
    };

    enum class GlobalIlluminationDebugView : uint8_t
    {
        None,
        IrradianceOnly,
        ContributionOnly,
        CoverageConfidence,
        EnabledDisabledComparison,
        DiagnosticIntensity,
        TracedRays,
        HitDistance,
        ReconstructedNormals,
        ReconstructedMaterials,
        ClipmapLevelProbeIdentity,
        ProbeClassificationRelocation,
        ProbeDepthVisibilityMoments,
        ProbeRadianceSh,
        CurrentPreviousIrradiance,
        TemporalRejectionReason,
        ProbeAgeUpdateState,
        ProbeConfidenceVariance,
        RawGatheredHalfResolution,
        ProbeUpdateWorkload,
        CachedProbeIrradiance,
        GatheredIrradianceCoverage,
        TemporalFilteredIrradiance,
        DirectOnly,
        TemporalHistoryWeight,
        Count,

        // Source-compatible aliases for old configuration files and tools.
        SelectedSurfelId = ClipmapLevelProbeIdentity,
        SurfelWorldPosition = ProbeClassificationRelocation,
        SurfelWorldNormal = ProbeDepthVisibilityMoments,
        RawShCoefficients = ProbeRadianceSh,
        AllocationUpdateAge = ProbeAgeUpdateState,
        VarianceConfidence = ProbeConfidenceVariance,
        RawHalfResolution = RawGatheredHalfResolution,
        RawTracedIndirectRadiance = ProbeUpdateWorkload,
        CachedSurfelRadiance = CachedProbeIrradiance,
        GatheredIrradiance = GatheredIrradianceCoverage,
        TemporalResult = TemporalFilteredIrradiance
    };

    struct GlobalIlluminationDebugViewInfo
    {
        GlobalIlluminationDebugView view = GlobalIlluminationDebugView::None;
        std::string_view name;
        std::string_view configName;
        std::string_view tooltip;
        std::string_view legend;
        std::string_view unitsAndRange;
        std::string_view sourcePass;
        std::string_view sourceResource;
        bool filtered = false;
        bool radiometric = false;
    };

    struct GlobalIlluminationPresetInfo
    {
        GlobalIlluminationQuality quality = GlobalIlluminationQuality::Off;
        std::string_view name;
        std::string_view description;
        float gpuTimeTargetMilliseconds = 0.0f;
        uint64_t memoryLimitBytes = 0;
    };

    struct GlobalIlluminationLimits
    {
        uint32_t maxSurfels = 131072;
        uint32_t hashBucketCount = 65536;
        uint32_t maxSurfelUpdates = 4096;
        uint32_t probeClipmapCount = 3;
        uint32_t probeResolution = 16;
        uint32_t maxProbeUpdates = 512;
        uint32_t rayBudget = 12288;
    };

    struct GlobalIlluminationConfiguration
    {
        bool enabled = false;
        GlobalIlluminationQuality quality = GlobalIlluminationQuality::Medium;
        GlobalIlluminationDebugView debugView =
            GlobalIlluminationDebugView::None;
        GlobalIlluminationLimits limits = {};
        // Explicit quality contracts. The time target is reporting/tuning
        // metadata; the memory limit is a hard graph-contribution ceiling.
        float gpuTimeTargetMilliseconds = 1.2f;
        uint64_t memoryLimitBytes = 128ull * 1024ull * 1024ull;
        // GI evaluation and temporal reconstruction share this resolution.
        // Two is half resolution; four is quarter resolution.
        uint32_t evaluationDivisor = 4;
        // World-space edge length of a surfel hash cell, in scene units. Smaller
        // cells give finer placement at higher memory/ray cost.
        float surfelCellSize = 0.30f;
        // Applied only by the DiagnosticIntensity view. Normal rendering always
        // uses unit intensity so this control cannot become an accidental
        // compensation for a weak or broken GI signal.
        float diagnosticIntensity = 1.0f;
        // Fixed display exposure for radiometric GI diagnostics. This never
        // changes from frame statistics and therefore cannot create false
        // flicker in a stable cached/filtered view.
        float debugExposure = 1.0f;
        // Diagnostic isolation control. Zero keeps normal continuous updates;
        // otherwise allocation and SH updates stop after this many frames.
        uint32_t freezeAfterFrames = 0;
        // Optional local-detail layer. The production path is probe-primary;
        // surfels stay off unless an acceptance capture proves incremental value.
        bool surfelDetail = false;
        // Hierarchical diffuse GI. Both stages are fixed-budget and use the
        // existing GPU scene/TLAS; they never allocate after graph creation.
        bool multiBounce = false;
        bool probeFallback = true;
        bool asyncCompute = true;
        bool diagnosticsReadback = false;
    };

    struct GlobalIlluminationCapabilities
    {
        RayTracingCapabilities rayTracing = {};
        bool storageImages = true;
        bool indirectDispatch = true;
        bool asyncCompute = false;

        [[nodiscard]] bool supported() const noexcept
        {
            return rayTracing.supportsInlineSceneQueries() && storageImages &&
                indirectDispatch;
        }
    };

    // 128-byte persistent surfel. The last three vec4s hold the per-channel
    // L1 spherical-harmonic diffuse irradiance (4 coefficients each), so no
    // separate SH buffer or extra binding is required.
    struct alignas(16) GpuRadianceSurfel
    {
        glm::vec4 positionRadius = {};      // xyz world position, w support radius
        glm::vec4 normalConfidence = {};    // xyz geometric normal, w confidence
        glm::vec4 radianceVariance = {};    // x mean luma, y mean luma^2, z samples, w age
        glm::uvec4 keyMaterialAgeFlags = {};// x cell key, y material id, z age frames, w flags
        glm::uvec4 instanceGeneration = {}; // x instance id, y generation, z spawn frame, w scene gen
        glm::vec4 shR = {};                 // L1 SH, red channel   (4 coefficients)
        glm::vec4 shG = {};                 // L1 SH, green channel (4 coefficients)
        glm::vec4 shB = {};                 // L1 SH, blue channel  (4 coefficients)
    };

    struct alignas(16) GpuVisibilityMoments
    {
        glm::vec4 meanMeanSquare = {};
        glm::vec4 irradianceSampleCount = {};
    };

    struct alignas(16) GpuSurfelCacheKey
    {
        glm::ivec3 quantizedCell = {};
        uint32_t normalCone = 0;
        uint32_t instanceId = UINT32_MAX;
        uint32_t instanceGeneration = 0;
        uint32_t materialId = UINT32_MAX;
        uint32_t padding = 0;
    };

    struct alignas(16) GpuProbeData
    {
        glm::vec4 shR = {};
        glm::vec4 shG = {};
        glm::vec4 shB = {};
        glm::vec4 depthMoments = {};
        glm::vec4 axisDepth0 = {};
        glm::vec4 axisDepth1 = {};
        glm::vec4 positionConfidence = {};
        glm::uvec4 clipmapAgeFlags = {};
    };

    struct alignas(16) GpuGiUpdateCommand
    {
        uint32_t cacheIndex = UINT32_MAX;
        uint32_t instanceId = UINT32_MAX;
        uint32_t instanceGeneration = 0;
        uint32_t rayOffset = 0;
        uint32_t rayCount = 0;
        float priority = 0.0f;
        uint32_t flags = 0;
        uint32_t padding = 0;
    };

    struct alignas(16) GpuGiTraceProbe
    {
        // physical index, expected world-cell key, stored world-cell key, flags
        glm::uvec4 state = {};
        glm::vec4 shR = {};
        glm::vec4 shG = {};
        glm::vec4 shB = {};
        glm::vec4 depthMoments = {};
    };

    struct alignas(16) GpuGiTracePixel
    {
        // selected probe, active field, sweep progress, temporal rejection
        glm::uvec4 state = {};
        glm::vec4 shR = {};
        glm::vec4 shG = {};
        glm::vec4 shB = {};
        glm::vec4 depthMoments = {};
        // gathered irradiance RGB, gather confidence
        glm::vec4 gathered = {};
        // temporally reconstructed input/history/output
        glm::vec4 temporalCurrent = {};
        glm::vec4 temporalHistory = {};
        glm::vec4 temporalOutput = {};
        // direct visibility, applied history weight, history confidence,
        // temporally filtered direct visibility
        glm::vec4 temporalState = {};
        // Eight trilinear contributors from the selected fine clipmap level.
        std::array<GpuGiTraceProbe, 8> probes = {};
    };

    struct alignas(16) GpuGiDiagnostics
    {
        uint32_t surfelOccupancy = 0;
        uint32_t hashCollisions = 0;
        uint32_t allocationFailures = 0;
        uint32_t surfelUpdateCount = 0;
        uint32_t irradianceLuminanceFixed = 0;
        uint32_t maxIrradianceBits = 0;
        uint32_t rejectedGathers = 0;
        uint32_t rayBudgetUsed = 0;
        uint32_t averageConfidenceBits = 0;
        uint32_t overflowFlags = 0;
        uint32_t historyEpoch = 0;
        uint32_t padding = 0;
        uint32_t tracedRays = 0;
        uint32_t hits = 0;
        uint32_t misses = 0;
        uint32_t alphaRejections = 0;
        uint32_t staleSurfels = 0;
        uint32_t invalidMappings = 0;
        uint32_t selfHitRejections = 0;
        uint32_t hitDistanceBits = 0;
        uint32_t validPixelCount = 0;
        uint32_t coverageFixed = 0;
        uint32_t referenceErrorFixed = 0;
        uint32_t referenceCount = 0;
        uint32_t irradianceRedFixed = 0;
        uint32_t irradianceGreenFixed = 0;
        uint32_t irradianceBlueFixed = 0;
        uint32_t temporalRejectedPixels = 0;
        uint32_t probeOccupancy = 0;
        uint32_t probeUpdateCount = 0;
        uint32_t feedbackSamples = 0;
        uint32_t probeFallbackPixels = 0;
        std::array<GpuGiTracePixel, 3> tracePixels = {};
    };

    struct alignas(16) GpuGiHitRecord
    {
        glm::vec4 positionDistance = {};
        glm::vec4 geometricNormalFlags = {};
        glm::vec4 shadingNormalFrontFace = {};
        glm::vec4 uvMaterial = {};
        glm::vec4 debugRadiance = {};
        glm::uvec4 indices = {};
        glm::vec4 barycentricsError = {};
        glm::uvec4 generationFlags = {};
    };

    // 128-byte constant block shared by every GI compute pass. DX12 binds it as
    // a constant buffer (b0); Vulkan supplies it as a push constant (the size is
    // exactly the guaranteed 128-byte minimum, so do not grow it past 32 words).
    struct alignas(16) GpuGiTraceConstants
    {
        uint32_t frameIndex = 0;
        uint32_t sceneGeneration = 0;
        uint32_t geometryCount = 0;
        uint32_t instanceCount = 0;
        uint32_t materialCount = 0;
        uint32_t textureCount = 0;
        uint32_t samplerCount = 0;
        uint32_t maxUpdates = 0;
        uint32_t rayBudget = 0;
        uint32_t raysPerSurfel = 1;
        uint32_t debugView = 0;
        uint32_t environmentEnabled = 0;
        float environmentIntensity = 1.0f;
        float rayMinDistance = 0.0001f;
        float rayMaxDistance = 10000.0f;
        uint32_t maxSurfels = 0;
        // --- persistent surfel-cache parameters ---
        float cellSize = 0.25f;
        float invCellSize = 4.0f;
        uint32_t hashBucketCount = 0;
        uint32_t candidatesPerCell = 4;
        float gatherRadiusScale = 1.5f;
        float normalThreshold = 0.5f;      // cos of max angle for compatible surfels
        float planeThreshold = 0.15f;      // world-space plane distance tolerance
        uint32_t maxSurfelAge = 256;       // frames without a visible touch before eviction
        uint32_t reducedWidth = 0;
        uint32_t reducedHeight = 0;
        uint32_t evaluationDivisor = 2;
        uint32_t feedbackEnabled = 0;
        float confidenceBlend = 1024.0f;  // packed runtime probe-update budget
        uint32_t emissiveInstanceIndex = UINT32_MAX;
        uint32_t giFlags = 0;
        uint32_t freezeAfterFrame = 0;
    };

    struct GlobalIlluminationRuntimeStatistics
    {
        GpuGiDiagnostics counters = {};
        std::array<double, 8> passGpuMilliseconds = {};
        double inclusiveGpuMilliseconds = 0.0;
        uint64_t allocatedBytes = 0;
        uint64_t probeBytes = 0;
        uint32_t configuredRayBudget = 0;
        uint32_t configuredProbeUpdates = 0;
        GlobalIlluminationQuality quality = GlobalIlluminationQuality::Off;
        bool hardwareSupported = false;
        bool active = false;
        bool memoryLimitExceeded = false;
        bool historyValid = false;
        uint64_t historyEpoch = 0;
    };

    struct GlobalIlluminationGraphResources
    {
        GraphResourceId surfaceData = InvalidGraphResourceId;
        GraphResourceId surfels = InvalidGraphResourceId;
        GraphResourceId visibilityMoments = InvalidGraphResourceId;
        GraphResourceId hashBuckets = InvalidGraphResourceId;
        GraphResourceId probes = InvalidGraphResourceId;
        GraphResourceId probeStaging = InvalidGraphResourceId;
        GraphResourceId allocationQueue = InvalidGraphResourceId;
        GraphResourceId priorityQueue = InvalidGraphResourceId;
        GraphResourceId indirectArguments = InvalidGraphResourceId;
        GraphResourceId counters = InvalidGraphResourceId;
        GraphResourceId diagnostics = InvalidGraphResourceId;
        GraphResourceId diagnosticsReadback = InvalidGraphResourceId;
        GraphResourceId residualInterface = InvalidGraphResourceId;
        GraphResourceId hitRecords = InvalidGraphResourceId;
        GraphResourceId rawIrradiance = InvalidGraphResourceId;
        GraphResourceId resolvedIrradiance = InvalidGraphResourceId;
    };

    struct GlobalIlluminationGraphPasses
    {
        GraphNodeId cacheInitialization = InvalidGraphNodeId;
        GraphNodeId surfacePreparation = InvalidGraphNodeId;
        GraphNodeId surfelAllocation = InvalidGraphNodeId;
        GraphNodeId prioritizeUpdates = InvalidGraphNodeId;
        GraphNodeId traceSurfelUpdates = InvalidGraphNodeId;
        GraphNodeId preserveProbes = InvalidGraphNodeId;
        GraphNodeId updateProbes = InvalidGraphNodeId;
        GraphNodeId evaluate = InvalidGraphNodeId;
        GraphNodeId temporalReconstruction = InvalidGraphNodeId;
    };

    // Inputs owned by the composing render path. GI only declares a reusable
    // subgraph into the caller's builder; it never clears, compiles, or executes
    // the renderer's frame graph.
    struct GlobalIlluminationGraphInputs
    {
        GraphResourceId sceneDepth = InvalidGraphResourceId;
        GraphResourceId surfaceAttributes = InvalidGraphResourceId;
        GraphResourceId rayTracingSceneToken = InvalidGraphResourceId;
        RenderExtent renderExtent = {};
        bool asyncComputeEnabled = false;
    };

    struct GlobalIlluminationGraphOutputs
    {
        GraphResourceId resolvedDiffuseIrradiance = InvalidGraphResourceId;

        [[nodiscard]] bool valid() const noexcept
        {
            return resolvedDiffuseIrradiance != InvalidGraphResourceId;
        }
    };

    class GlobalIlluminationSystem final
    {
    public:
        void configure(const GlobalIlluminationConfiguration& config);
        void setCapabilities(const GlobalIlluminationCapabilities& capabilities);

        [[nodiscard]] bool active() const noexcept;
        [[nodiscard]] bool requested() const noexcept { return m_config.enabled; }
        [[nodiscard]] const GlobalIlluminationConfiguration& configuration()
            const noexcept { return m_config; }
        [[nodiscard]] const GlobalIlluminationCapabilities& capabilities()
            const noexcept { return m_capabilities; }
        [[nodiscard]] const GlobalIlluminationRuntimeStatistics& statistics()
            const noexcept { return m_statistics; }

        void invalidateHistory() noexcept;

        [[nodiscard]] GlobalIlluminationGraphOutputs contributePasses(
            FrameGraphBuilder& builder,
            const GlobalIlluminationGraphInputs& inputs);

        [[nodiscard]] const GlobalIlluminationGraphResources& resources()
            const noexcept { return m_resources; }
        [[nodiscard]] const GlobalIlluminationGraphPasses& passes()
            const noexcept { return m_passes; }

        static GlobalIlluminationLimits sanitizeLimits(
            GlobalIlluminationLimits limits) noexcept;

        [[nodiscard]] static uint64_t estimateAllocatedBytes(
            const GlobalIlluminationConfiguration& config,
            RenderExtent renderExtent) noexcept;

    private:
        GlobalIlluminationConfiguration m_config = {};
        GlobalIlluminationCapabilities m_capabilities = {};
        GlobalIlluminationRuntimeStatistics m_statistics = {};
        GlobalIlluminationGraphResources m_resources = {};
        GlobalIlluminationGraphPasses m_passes = {};
    };

    static_assert(sizeof(GpuRadianceSurfel) == 128);
    static_assert(sizeof(GpuVisibilityMoments) == 32);
    static_assert(sizeof(GpuSurfelCacheKey) == 32);
    static_assert(sizeof(GpuProbeData) == 128);
    static_assert(sizeof(GpuGiUpdateCommand) == 32);
    static_assert(sizeof(GpuGiTraceProbe) == 80);
    static_assert(sizeof(GpuGiTracePixel) == 800);
    static_assert(sizeof(GpuGiDiagnostics) == 2528);
    static_assert(sizeof(GpuGiHitRecord) == 128);
    static_assert(sizeof(GpuGiTraceConstants) == 128);

    [[nodiscard]] std::span<const GlobalIlluminationDebugViewInfo>
        globalIlluminationDebugViews() noexcept;
    [[nodiscard]] const GlobalIlluminationDebugViewInfo&
        globalIlluminationDebugViewInfo(
            GlobalIlluminationDebugView view) noexcept;
    [[nodiscard]] std::span<const GlobalIlluminationPresetInfo>
        globalIlluminationPresets() noexcept;
    [[nodiscard]] const GlobalIlluminationPresetInfo&
        globalIlluminationPresetInfo(
            GlobalIlluminationQuality quality) noexcept;
    [[nodiscard]] GlobalIlluminationConfiguration
        globalIlluminationPreset(
            GlobalIlluminationQuality quality,
            const GlobalIlluminationConfiguration& base = {}) noexcept;
}
