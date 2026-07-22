#include "ic/common/ic_pch.h"
#include "ic/renderer/global_illumination/global_illumination.h"
#include "ic/renderer/frame_graph/frame_graph_builder.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace ic
{
    namespace
    {
        constexpr uint32_t kMaxSurfels = 4u * 1024u * 1024u;
        constexpr uint32_t kMaxHashBuckets = 4u * 1024u * 1024u;
        constexpr uint32_t kMaxUpdates = 262144u;

        uint64_t checkedBytes(uint64_t count, uint64_t stride) noexcept
        {
            if (stride && count > std::numeric_limits<uint64_t>::max() / stride)
                return std::numeric_limits<uint64_t>::max();
            return std::max<uint64_t>(count * stride, 16u);
        }

        uint32_t floatBits(float value) noexcept
        {
            uint32_t bits = 0;
            static_assert(sizeof(bits) == sizeof(value));
            std::memcpy(&bits, &value, sizeof(bits));
            return bits;
        }

        BufferDesc storageBuffer(
            uint64_t size, const char* name,
            BufferUsageFlags extra = BufferUsageFlags::None)
        {
            return { .size = size,
                .usage = BufferUsageFlags::Storage |
                    BufferUsageFlags::TransferDst | extra,
                .memoryUsage = ResourceMemoryUsage::GpuOnly,
                .mappedAtCreation = false, .debugName = name };
        }

        constexpr uint64_t mib(uint64_t value) noexcept
        {
            return value * 1024u * 1024u;
        }

        constexpr std::array<GlobalIlluminationDebugViewInfo, 25>
            kDebugViews{{
            { GlobalIlluminationDebugView::None, "Final lighting", "none",
                "Normal clustered-forward lighting with resolved diffuse GI.",
                "RGB: final HDR lighting", "scene-linear HDR", "Forward.Opaque",
                "HDR color + GI.ResolvedDiffuseIrradiance", true, true },
            { GlobalIlluminationDebugView::IrradianceOnly, "Resolved irradiance", "irradiance_only",
                "Temporally reconstructed diffuse irradiance before material response.",
                "RGB: irradiance", "scene-linear radiance, fixed exposure", "Forward.Opaque",
                "GI.ResolvedDiffuseIrradiance", true, true },
            { GlobalIlluminationDebugView::ContributionOnly, "GI contribution only", "contribution_only",
                "Diffuse GI after albedo and Lambertian response, without direct lighting.",
                "RGB: applied GI", "scene-linear radiance, fixed exposure", "Forward.Opaque",
                "GI.ResolvedDiffuseIrradiance + GBuffer material", true, true },
            { GlobalIlluminationDebugView::CoverageConfidence, "Coverage / confidence", "coverage_confidence",
                "Resolved gather coverage and confidence.", "R: missing coverage, G: confidence, B: covered",
                "normalized [0,1]", "GI.TemporalReconstruction", "GI.ResolvedDiffuseIrradiance.a", true, false },
            { GlobalIlluminationDebugView::EnabledDisabledComparison, "GI off / on split", "enabled_disabled",
                "Left half is direct-only; right half adds GI from the same resolved frame.",
                "left: GI off, right: GI on", "scene-linear HDR", "Forward.Opaque",
                "direct lighting + GI.ResolvedDiffuseIrradiance", true, true },
            { GlobalIlluminationDebugView::DiagnosticIntensity, "GI diagnostic intensity", "diagnostic_intensity",
                "Contribution-only view multiplied by the diagnostic intensity control.",
                "RGB: scaled GI contribution", "scene-linear radiance, manual multiplier", "Forward.Opaque",
                "GI.ResolvedDiffuseIrradiance", true, true },
            { GlobalIlluminationDebugView::TracedRays, "Probe ray hits / misses", "traced_rays",
                "Hit/miss ratio stored by the selected probe's most recent scheduled ray batch.",
                "G: hit ratio, B: miss ratio", "normalized [0,1]", "GI.UpdateProbes/GI.EvaluateDiffuse",
                "completed GI probe snapshot depth moments", true, false },
            { GlobalIlluminationDebugView::HitDistance, "Ray hit distance", "hit_distance",
                "Mean distance stored by the selected probe's most recent ray batch.", "black-near to white-far", "normalized [0, probe ray max]",
                "GI.UpdateProbes/GI.EvaluateDiffuse", "completed GI probe snapshot depth moments", true, false },
            { GlobalIlluminationDebugView::ReconstructedNormals, "Evaluation surface normal", "reconstructed_normals",
                "World-space receiver normal reconstructed for GI evaluation.", "RGB: normal * 0.5 + 0.5",
                "unit vector encoded [0,1]", "GI.SurfacePreparation/GI.EvaluateDiffuse", "GI.SurfaceData", true, false },
            { GlobalIlluminationDebugView::ReconstructedMaterials, "Evaluation material identity", "reconstructed_materials",
                "Deterministic color hash of the receiver material used for GI evaluation.", "RGB: material identity",
                "categorical", "GI.SurfacePreparation/GI.EvaluateDiffuse", "GI.SurfaceData", true, false },
            { GlobalIlluminationDebugView::ClipmapLevelProbeIdentity, "Clipmap / probe identity", "clipmap_probe_identity",
                "Selected immutable probe identity; hue is identity and brightness is clipmap level.",
                "RGB: stable identity hash, brightness: fine to coarse", "categorical", "GI.EvaluateDiffuse",
                "completed GI probe snapshot", true, false },
            { GlobalIlluminationDebugView::ProbeClassificationRelocation, "Probe classification / relocation", "probe_classification_relocation",
                "Classification state and normalized relocation of the selected completed probe.",
                "R: relocation, G: confidence, B: classified", "normalized [0,1]", "GI.EvaluateDiffuse",
                "completed GI probe snapshot", true, false },
            { GlobalIlluminationDebugView::ProbeDepthVisibilityMoments, "Depth / visibility moments", "probe_depth_visibility_moments",
                "Selected-probe depth mean/deviation and the gather's evaluated visibility.",
                "R: selected mean depth, G: selected depth sigma, B: gathered visibility", "normalized [0,1]", "GI.EvaluateDiffuse",
                "completed GI probe snapshot", true, false },
            { GlobalIlluminationDebugView::ProbeRadianceSh, "Probe radiance / SH DC", "probe_radiance_sh",
                "Weighted DC radiance coefficient from the completed probes participating in the gather.",
                "RGB: SH DC radiance", "scene-linear radiance, fixed exposure", "GI.EvaluateDiffuse",
                "completed GI probe snapshot", true, true },
            { GlobalIlluminationDebugView::CurrentPreviousIrradiance, "Current / history irradiance", "current_previous_irradiance",
                "Split comparison before temporal blending.", "left: current gather, right: previous resolved history",
                "scene-linear radiance, fixed exposure", "GI.TemporalReconstruction", "GI.RawDiffuseIrradiance + previous GI.ResolvedDiffuseIrradiance", true, true },
            { GlobalIlluminationDebugView::TemporalRejectionReason, "Temporal rejection reason", "temporal_rejection_reason",
                "Why temporal history was rejected for each pixel.", "green: accepted, red: depth, blue: normal/material, yellow: invalid/cut",
                "categorical", "GI.TemporalReconstruction", "temporal validation state", true, false },
            { GlobalIlluminationDebugView::ProbeAgeUpdateState, "Probe age / update state", "probe_age_update_state",
                "Age and resident/update state of the selected completed probe.",
                "R: age/512, G: updated this sweep, B: resident", "normalized [0,1]", "GI.EvaluateDiffuse",
                "completed GI probe snapshot", true, false },
            { GlobalIlluminationDebugView::ProbeConfidenceVariance, "Probe confidence / variance", "probe_confidence_variance",
                "Radiance variance, confidence, and visibility of the selected completed probe.",
                "R: variance, G: confidence, B: visibility", "normalized [0,1]", "GI.EvaluateDiffuse",
                "completed GI probe snapshot", true, false },
            { GlobalIlluminationDebugView::RawGatheredHalfResolution, "Raw gathered irradiance", "raw_gathered_half_resolution",
                "Unfiltered evaluation-resolution gather, nearest-expanded to the display.",
                "RGB: raw gather, A: confidence", "scene-linear radiance, fixed exposure", "GI.TemporalReconstruction",
                "GI.RawDiffuseIrradiance", false, true },
            { GlobalIlluminationDebugView::ProbeUpdateWorkload, "Probe update workload", "probe_update_workload",
                "Shows whether the selected probe was updated this frame and its ray hit/miss ratio.",
                "R: updated, G: hit ratio, B: miss ratio", "normalized [0,1]", "GI.UpdateProbes/GI.EvaluateDiffuse",
                "completed GI probe snapshot", false, false },
            { GlobalIlluminationDebugView::CachedProbeIrradiance, "Cached probe irradiance", "cached_probe_irradiance",
                "Irradiance reconstructed directly from the selected completed probe cache.",
                "RGB: cached irradiance", "scene-linear irradiance, fixed exposure", "GI.EvaluateDiffuse",
                "completed GI probe snapshot", true, true },
            { GlobalIlluminationDebugView::GatheredIrradianceCoverage, "Gathered irradiance / coverage", "gathered_irradiance_coverage",
                "Weighted multi-probe gather and its coverage before temporal reconstruction.",
                "RGB: gathered irradiance, A: coverage", "scene-linear irradiance, fixed exposure", "GI.EvaluateDiffuse",
                "GI.RawDiffuseIrradiance", true, true },
            { GlobalIlluminationDebugView::TemporalFilteredIrradiance, "Temporal filtered irradiance", "temporal_filtered_irradiance",
                "The exact resolved irradiance consumed by final clustered-forward shading.",
                "RGB: resolved irradiance, A: confidence", "scene-linear irradiance, fixed exposure", "GI.TemporalReconstruction",
                "GI.ResolvedDiffuseIrradiance", true, true },
            { GlobalIlluminationDebugView::DirectOnly, "Direct lighting only", "direct_only",
                "Clustered-forward direct and emissive lighting with indirect lighting omitted.",
                "RGB: direct + emissive", "scene-linear HDR", "Forward.Opaque", "HDR color before indirect lighting", true, true },
            { GlobalIlluminationDebugView::TemporalHistoryWeight, "Temporal history weight", "temporal_history_weight",
                "History weight actually applied after all rejection tests.", "black: current only, white: history only",
                "normalized [0,1]", "GI.TemporalReconstruction", "GI.ResolvedDiffuseIrradiance debug output", true, false }
        }};

        constexpr std::array<GlobalIlluminationPresetInfo, 5> kPresets{{
            { GlobalIlluminationQuality::Off, "Off", "Disables diffuse GI and its RT/resource demand.", 0.0f, 0 },
            { GlobalIlluminationQuality::Low, "Low", "Two compact clipmaps; slower convergence and quarter-resolution evaluation.", 1.1f, mib(96) },
            { GlobalIlluminationQuality::Medium, "Medium", "Three clipmaps with balanced update and visibility budgets.", 1.2f, mib(128) },
            { GlobalIlluminationQuality::High, "High", "Production four-clipmap field with half-resolution evaluation.", 1.9f, mib(192) },
            { GlobalIlluminationQuality::Ultra, "Ultra", "Higher spatial density and update throughput; no experimental feedback features.", 2.5f, mib(320) }
        }};
    }

    std::span<const GlobalIlluminationDebugViewInfo>
        globalIlluminationDebugViews() noexcept { return kDebugViews; }

    const GlobalIlluminationDebugViewInfo& globalIlluminationDebugViewInfo(
        GlobalIlluminationDebugView view) noexcept
    {
        const size_t index = static_cast<size_t>(view);
        return kDebugViews[index < kDebugViews.size() ? index : 0u];
    }

    std::span<const GlobalIlluminationPresetInfo>
        globalIlluminationPresets() noexcept { return kPresets; }

    const GlobalIlluminationPresetInfo& globalIlluminationPresetInfo(
        GlobalIlluminationQuality quality) noexcept
    {
        const size_t index = static_cast<size_t>(quality);
        return kPresets[index < kPresets.size() ? index : 0u];
    }

    GlobalIlluminationConfiguration globalIlluminationPreset(
        GlobalIlluminationQuality quality,
        const GlobalIlluminationConfiguration& base) noexcept
    {
        GlobalIlluminationConfiguration result = base;
        result.quality = quality;
        result.enabled = quality != GlobalIlluminationQuality::Off;
        if (quality <= GlobalIlluminationQuality::Ultra)
        {
            const auto& info = globalIlluminationPresetInfo(quality);
            result.gpuTimeTargetMilliseconds =
                info.gpuTimeTargetMilliseconds;
            result.memoryLimitBytes = info.memoryLimitBytes;
        }
        result.multiBounce = false;
        result.probeFallback = true;
        result.surfelDetail = false;
        switch (quality)
        {
        case GlobalIlluminationQuality::Off:
            return result;
        case GlobalIlluminationQuality::Low:
            result.limits = { 65536, 32768, 1024, 2, 12, 256, 4096 };
            result.evaluationDivisor = 4;
            result.surfelCellSize = 0.35f;
            break;
        case GlobalIlluminationQuality::Medium:
            result.limits = { 131072, 65536, 4096, 3, 16, 512, 12288 };
            result.evaluationDivisor = 4;
            result.surfelCellSize = 0.30f;
            break;
        case GlobalIlluminationQuality::High:
            result.limits = { 262144, 131072, 8192, 4, 24, 1024, 32768 };
            result.evaluationDivisor = 2;
            result.surfelCellSize = 0.25f;
            break;
        case GlobalIlluminationQuality::Ultra:
            result.limits = { 524288, 262144, 16384, 4, 32, 2048, 65536 };
            result.evaluationDivisor = 2;
            result.surfelCellSize = 0.20f;
            break;
        case GlobalIlluminationQuality::Custom:
        case GlobalIlluminationQuality::Count:
            result.quality = GlobalIlluminationQuality::Custom;
            break;
        }
        return result;
    }

    GlobalIlluminationLimits GlobalIlluminationSystem::sanitizeLimits(
        GlobalIlluminationLimits limits) noexcept
    {
        limits.maxSurfels = std::clamp(limits.maxSurfels, 1u, kMaxSurfels);
        limits.hashBucketCount = std::clamp(
            limits.hashBucketCount, 1u, kMaxHashBuckets);
        limits.maxSurfelUpdates = std::clamp(
            limits.maxSurfelUpdates, 1u, kMaxUpdates);
        limits.probeClipmapCount = std::clamp(
            limits.probeClipmapCount, 1u, 8u);
        limits.probeResolution = std::clamp(
            limits.probeResolution, 2u, 64u);
        limits.maxProbeUpdates = std::clamp(
            limits.maxProbeUpdates, 1u, kMaxUpdates);
        limits.maxProbeUpdates = std::max(
            64u, (limits.maxProbeUpdates / 64u) * 64u);
        limits.rayBudget = std::clamp(
            limits.rayBudget, 1u, 16u * 1024u * 1024u);
        return limits;
    }

    void GlobalIlluminationSystem::configure(
        const GlobalIlluminationConfiguration& config)
    {
        GlobalIlluminationConfiguration sanitized = config;
        sanitized.limits = sanitizeLimits(sanitized.limits);
        sanitized.evaluationDivisor =
            sanitized.evaluationDivisor <= 2u ? 2u : 4u;
        sanitized.diagnosticIntensity = std::clamp(
            sanitized.diagnosticIntensity, 0.0f, 16.0f);
        sanitized.debugExposure = std::clamp(
            sanitized.debugExposure, 0.03125f, 32.0f);
        sanitized.gpuTimeTargetMilliseconds = std::clamp(
            sanitized.gpuTimeTargetMilliseconds, 0.0f, 1000.0f);
        const bool incompatible =
            m_config.limits.maxSurfels != sanitized.limits.maxSurfels ||
            m_config.limits.hashBucketCount != sanitized.limits.hashBucketCount ||
            m_config.limits.probeClipmapCount != sanitized.limits.probeClipmapCount ||
            m_config.limits.probeResolution != sanitized.limits.probeResolution ||
            m_config.evaluationDivisor != sanitized.evaluationDivisor;
        m_config = sanitized;
        m_statistics.quality = sanitized.enabled
            ? sanitized.quality : GlobalIlluminationQuality::Off;
        m_statistics.configuredRayBudget = sanitized.enabled
            ? sanitized.limits.rayBudget : 0u;
        m_statistics.configuredProbeUpdates = sanitized.enabled
            ? sanitized.limits.maxProbeUpdates : 0u;
        m_statistics.active = active();
        if (incompatible) invalidateHistory();
    }

    void GlobalIlluminationSystem::setCapabilities(
        const GlobalIlluminationCapabilities& capabilities)
    {
        if (m_capabilities.supported() != capabilities.supported())
            invalidateHistory();
        m_capabilities = capabilities;
        m_statistics.hardwareSupported = m_statistics.hardwareSupported ||
            capabilities.rayTracing.accelerationStructures &&
            capabilities.rayTracing.inlineRayQueries &&
            capabilities.storageImages && capabilities.indirectDispatch;
        m_statistics.active = active();
    }

    bool GlobalIlluminationSystem::active() const noexcept
    {
        return m_config.enabled && m_capabilities.supported();
    }

    void GlobalIlluminationSystem::invalidateHistory() noexcept
    {
        ++m_statistics.historyEpoch;
        m_statistics.historyValid = false;
    }

    uint64_t GlobalIlluminationSystem::estimateAllocatedBytes(
        const GlobalIlluminationConfiguration& input,
        RenderExtent renderExtent) noexcept
    {
        if (!input.enabled)
            return 0;
        const GlobalIlluminationLimits limits = sanitizeLimits(input.limits);
        const uint32_t divisor = input.evaluationDivisor <= 2u ? 2u : 4u;
        const uint64_t width = std::max(1u,
            (renderExtent.width + divisor - 1u) / divisor);
        const uint64_t height = std::max(1u,
            (renderExtent.height + divisor - 1u) / divisor);
        const uint64_t pixels = width * height;
        const uint64_t fullPixels = std::max<uint64_t>(1u, renderExtent.width) *
            std::max<uint64_t>(1u, renderExtent.height);
        const uint64_t surfels = input.surfelDetail ? limits.maxSurfels : 1u;
        const uint64_t updates = input.surfelDetail ? limits.maxSurfelUpdates : 1u;
        const uint64_t buckets = input.surfelDetail
            ? ((static_cast<uint64_t>(limits.hashBucketCount) + 63u) & ~63ull)
            : 64u;
        const uint64_t resolution = limits.probeResolution;
        const uint64_t probes = resolution * resolution * resolution *
            limits.probeClipmapCount;

        uint64_t bytes = 0;
        bytes += pixels * 32u; // reconstructed surface data
        bytes += surfels * (sizeof(GpuRadianceSurfel) +
            sizeof(GpuVisibilityMoments) + sizeof(GpuGiHitRecord));
        bytes += buckets * sizeof(glm::uvec4);
        bytes += probes * sizeof(GpuProbeData) * 2u; // immutable source/destination
        bytes += updates * (sizeof(uint32_t) + sizeof(GpuGiUpdateCommand));
        bytes += 64u + 3u * sizeof(uint32_t) + sizeof(GpuGiDiagnostics) * 2u + 64u;
        bytes += pixels * 16u;       // raw RGBA32F
        bytes += fullPixels * 16u * 2u; // resolved history ping-pong
        if (input.diagnosticsReadback)
            bytes += sizeof(GpuGiDiagnostics);
        return bytes;
    }

    GlobalIlluminationGraphOutputs GlobalIlluminationSystem::contributePasses(
        FrameGraphBuilder& builder,
        const GlobalIlluminationGraphInputs& inputs)
    {
        const GraphResourceId sceneDepth = inputs.sceneDepth;
        const GraphResourceId surfaceAttributes = inputs.surfaceAttributes;
        const RenderExtent renderExtent = inputs.renderExtent;
        const GraphResourceId rayTracingSceneToken =
            inputs.rayTracingSceneToken;
        m_resources = {};
        m_passes = {};
        m_statistics.active = active();
        m_statistics.hardwareSupported = m_statistics.hardwareSupported ||
            (m_capabilities.rayTracing.accelerationStructures &&
             m_capabilities.rayTracing.inlineRayQueries &&
             m_capabilities.storageImages &&
             m_capabilities.indirectDispatch);
        m_statistics.quality = m_config.enabled
            ? m_config.quality : GlobalIlluminationQuality::Off;
        m_statistics.configuredRayBudget = m_config.enabled
            ? m_config.limits.rayBudget : 0u;
        m_statistics.configuredProbeUpdates = m_config.enabled
            ? m_config.limits.maxProbeUpdates : 0u;
        const uint64_t estimatedBytes = active()
            ? estimateAllocatedBytes(m_config, renderExtent) : 0u;
        const bool withinMemoryLimit = m_config.memoryLimitBytes == 0u ||
            estimatedBytes <= m_config.memoryLimitBytes;
        m_statistics.memoryLimitExceeded = active() && !withinMemoryLimit;
        m_statistics.active = active() && withinMemoryLimit;
        m_statistics.allocatedBytes = withinMemoryLimit ? estimatedBytes : 0u;
        const uint64_t statisticsProbeResolution =
            m_config.limits.probeResolution;
        m_statistics.probeBytes = active() && withinMemoryLimit
            ? statisticsProbeResolution * statisticsProbeResolution *
                statisticsProbeResolution *
                m_config.limits.probeClipmapCount * sizeof(GpuProbeData) * 2u
            : 0u;
        if (!active() || !withinMemoryLimit ||
            sceneDepth == InvalidGraphResourceId ||
            surfaceAttributes == InvalidGraphResourceId)
            return {};

        const auto limits = m_config.limits;
        const uint32_t divisor = m_config.evaluationDivisor;
        const uint32_t width = std::max(
            1u, (renderExtent.width + divisor - 1u) / divisor);
        const uint32_t height = std::max(
            1u, (renderExtent.height + divisor - 1u) / divisor);
        const uint64_t pixels = static_cast<uint64_t>(width) * height;
        const uint32_t surfelCount = m_config.surfelDetail
            ? limits.maxSurfels : 1u;
        const uint32_t surfelUpdateCount = m_config.surfelDetail
            ? limits.maxSurfelUpdates : 1u;
        const uint32_t hashBucketCount = m_config.surfelDetail
            ? limits.hashBucketCount : 1u;

        m_resources.surfaceData = builder.createBuffer(storageBuffer(
            checkedBytes(pixels, 32u), "GI.SurfaceData"));
        m_resources.surfels = builder.createPersistentBuffer(storageBuffer(
            checkedBytes(surfelCount, sizeof(GpuRadianceSurfel)),
            "GI.Surfels"));
        m_resources.visibilityMoments = builder.createPersistentBuffer(storageBuffer(
            checkedBytes(surfelCount, sizeof(GpuVisibilityMoments)),
            "GI.VisibilityMoments"));
        const uint64_t paddedHashBucketCount =
            (static_cast<uint64_t>(hashBucketCount) + 63u) & ~63ull;
        m_resources.hashBuckets = builder.createPersistentBuffer(storageBuffer(
            checkedBytes(paddedHashBucketCount, sizeof(glm::uvec4)),
            "GI.HashBuckets", BufferUsageFlags::TransferDst));
        const uint64_t probeResolution = limits.probeResolution;
        const uint64_t probesPerClipmap = probeResolution * probeResolution *
            probeResolution;
        const uint64_t probeCount = probesPerClipmap *
            limits.probeClipmapCount;
        m_resources.probes = builder.createPersistentBuffer(storageBuffer(
            checkedBytes(probeCount, sizeof(GpuProbeData)),
            "GI.ClipmappedProbes"));
        m_resources.probeStaging = builder.createPersistentBuffer(storageBuffer(
            checkedBytes(probeCount, sizeof(GpuProbeData)),
            "GI.ClipmappedProbeStaging"));
        m_resources.allocationQueue = builder.createBuffer(storageBuffer(
            checkedBytes(surfelUpdateCount, sizeof(uint32_t)),
            "GI.AllocationQueue"));
        m_resources.priorityQueue = builder.createBuffer(storageBuffer(
            checkedBytes(surfelUpdateCount, sizeof(GpuGiUpdateCommand)),
            "GI.PriorityQueue"));
        m_resources.indirectArguments = builder.createBuffer(storageBuffer(
            3u * sizeof(uint32_t), "GI.IndirectArguments",
            BufferUsageFlags::Indirect));
        m_resources.counters = builder.createBuffer(storageBuffer(
            64u, "GI.Counters"));
        m_resources.diagnostics = builder.createHistoryBuffer(storageBuffer(
            sizeof(GpuGiDiagnostics), "GI.Diagnostics",
            m_config.diagnosticsReadback ? BufferUsageFlags::TransferSrc
                                         : BufferUsageFlags::None));
        if (m_config.diagnosticsReadback)
        {
            m_resources.diagnosticsReadback = builder.createBuffer({
                .size = sizeof(GpuGiDiagnostics),
                .usage = BufferUsageFlags::TransferDst,
                .memoryUsage = ResourceMemoryUsage::GpuToCpu,
                .mappedAtCreation = true,
                .debugName = "GI.DiagnosticsReadback" });
            builder.setResourceSemantic(m_resources.diagnosticsReadback,
                GraphResourceSemantic::GiDiagnosticsReadback);
        }
        m_resources.residualInterface = builder.createPersistentBuffer(
            storageBuffer(64u, "GI.ResidualInterface",
                BufferUsageFlags::TransferDst));
        m_resources.hitRecords = builder.createPersistentBuffer(storageBuffer(
            checkedBytes(surfelCount, sizeof(GpuGiHitRecord)),
            "GI.SurfelHitRecords"));

        const TextureDesc rawDesc{ .width = width, .height = height,
            .depth = 1, .mipLevels = 1, .arrayLayers = 1,
            .format = TextureFormat::RGBA32_Float,
            .usage = TextureUsageFlags::Sampled | TextureUsageFlags::Storage,
            .memoryUsage = ResourceMemoryUsage::GpuOnly,
            .debugName = "GI.RawDiffuseIrradiance" };
        m_resources.rawIrradiance = builder.createTexture(rawDesc);
        TextureDesc historyDesc = rawDesc;
        historyDesc.width = renderExtent.width;
        historyDesc.height = renderExtent.height;
        historyDesc.debugName = "GI.ResolvedDiffuseIrradiance";
        m_resources.resolvedIrradiance = builder.createHistoryTexture(historyDesc);

        const std::array<std::pair<GraphResourceId, GraphResourceSemantic>, 15>
            semantics{{
                {m_resources.surfaceData, GraphResourceSemantic::GiSurfaceData},
                {m_resources.surfels, GraphResourceSemantic::GiSurfels},
                {m_resources.visibilityMoments, GraphResourceSemantic::GiVisibilityMoments},
                {m_resources.hashBuckets, GraphResourceSemantic::GiHashBuckets},
                {m_resources.probes, GraphResourceSemantic::GiProbes},
                {m_resources.probeStaging, GraphResourceSemantic::GiProbeStaging},
                {m_resources.allocationQueue, GraphResourceSemantic::GiAllocationQueue},
                {m_resources.priorityQueue, GraphResourceSemantic::GiPriorityQueue},
                {m_resources.indirectArguments, GraphResourceSemantic::GiIndirectArguments},
                {m_resources.counters, GraphResourceSemantic::GiCounters},
                {m_resources.diagnostics, GraphResourceSemantic::GiDiagnostics},
                {m_resources.residualInterface, GraphResourceSemantic::GiResidualInterface},
                {m_resources.hitRecords, GraphResourceSemantic::GiHitRecords},
                {m_resources.rawIrradiance, GraphResourceSemantic::GiRawIrradiance},
                {m_resources.resolvedIrradiance, GraphResourceSemantic::GiResolvedIrradiance}
            }};
        for (const auto& [resource, semantic] : semantics)
            builder.setResourceSemantic(resource, semantic);

        const QueueType queue = inputs.asyncComputeEnabled &&
            m_config.asyncCompute &&
            m_capabilities.asyncCompute ? QueueType::Compute : QueueType::Graphics;
        const uint32_t pixelGroups = static_cast<uint32_t>((pixels + 63u) / 64u);

        // Every GI pass receives the same parameter block; the backend expands it
        // into GpuGiTraceConstants. Order: [0] max surfel updates, [1] ray budget,
        // [2] baseline rays per surfel, [3] debug view, [4] max surfels,
        // [5] hash bucket count, [6] cell size (float bits), [7] packed hierarchy:
        // bits 0..3 divisor, 4..7 clipmap count, 8..14 probe resolution,
        // bit 15 multi-bounce feedback, bit 16 probe fallback, bit 17 optional
        // surfel detail. With bit 17 clear the DDGI field is the primary cache.
        const uint32_t cellBits = floatBits(std::max(m_config.surfelCellSize, 1.0e-3f));
        const uint32_t hierarchyBits = (divisor & 0xfu) |
            ((limits.probeClipmapCount & 0xfu) << 4u) |
            ((limits.probeResolution & 0x7fu) << 8u) |
            (m_config.multiBounce ? (1u << 15u) : 0u) |
            (m_config.probeFallback ? (1u << 16u) : 0u) |
            (m_config.surfelDetail ? (1u << 17u) : 0u);
        const uint32_t updateBits =
            (surfelUpdateCount & 0x3ffffu) |
            (((limits.maxProbeUpdates / 64u) & 0xfffu) << 18u);
#define IC_GI_USER_DATA \
    userData(updateBits, limits.rayBudget, 16u, \
        static_cast<uint32_t>(m_config.debugView), surfelCount, \
        hashBucketCount, cellBits, hierarchyBits)

        auto initialize = builder.addComputePass(
                "GI.CacheInitialization", QueueType::Graphics)
            .pipeline("gi_cache_initialize")
            .dispatch(static_cast<uint32_t>((std::max(
                paddedHashBucketCount, probeCount) + 63u) / 64u),
                1u, 1u)
            .IC_GI_USER_DATA
            .write(m_resources.hashBuckets, ResourceUsage::TransferDst)
            .write(m_resources.probes, ResourceUsage::TransferDst)
            .write(m_resources.probeStaging, ResourceUsage::TransferDst)
            .write(m_resources.residualInterface, ResourceUsage::TransferDst)
            .once();
        m_passes.cacheInitialization = initialize;

        auto surface = builder.addComputePass("GI.SurfacePreparation", queue)
            .pipeline("gi_surface_prepare")
            .dispatch(std::max(1u, pixelGroups), 1, 1)
            .IC_GI_USER_DATA
            .read(sceneDepth, ResourceUsage::SampledTexture)
            .read(surfaceAttributes, ResourceUsage::SampledTexture)
            .write(m_resources.surfaceData, ResourceUsage::StorageBuffer)
            .write(m_resources.counters, ResourceUsage::StorageBuffer)
            .write(m_resources.diagnostics, ResourceUsage::StorageBuffer)
            .read(m_resources.residualInterface, ResourceUsage::StorageBuffer)
            .asyncEligible();
        m_passes.surfacePreparation = surface;

        if (m_config.surfelDetail)
        {
            auto allocate = builder.addComputePass("GI.SurfelAllocation", queue)
                .pipeline("gi_surfel_allocate")
                .dispatch(std::max(1u, pixelGroups), 1, 1)
                .IC_GI_USER_DATA
                .read(m_resources.surfaceData, ResourceUsage::StorageBuffer)
                .write(m_resources.surfels, ResourceUsage::StorageBuffer)
                .write(m_resources.visibilityMoments, ResourceUsage::StorageBuffer)
                .write(m_resources.hashBuckets, ResourceUsage::StorageBuffer)
                .write(m_resources.allocationQueue, ResourceUsage::StorageBuffer)
                .write(m_resources.counters, ResourceUsage::StorageBuffer)
                .write(m_resources.residualInterface, ResourceUsage::StorageBuffer)
                .write(m_resources.diagnostics, ResourceUsage::StorageBuffer)
                .asyncEligible();
            m_passes.surfelAllocation = allocate;

            auto prioritize = builder.addComputePass("GI.PrioritizeUpdates", queue)
                .pipeline("gi_prioritize_updates")
                .dispatch(std::max(1u,
                    (limits.maxSurfelUpdates + 63u) / 64u), 1, 1)
                .IC_GI_USER_DATA
                .read(m_resources.allocationQueue, ResourceUsage::StorageBuffer)
                .read(m_resources.surfels, ResourceUsage::StorageBuffer)
                .write(m_resources.priorityQueue, ResourceUsage::StorageBuffer)
                .write(m_resources.indirectArguments, ResourceUsage::StorageBuffer)
                .write(m_resources.counters, ResourceUsage::StorageBuffer)
                .asyncEligible();
            m_passes.prioritizeUpdates = prioritize;

            auto trace = builder.addComputePass("GI.TraceSurfelUpdates", queue)
                .pipeline("gi_trace_surfel_updates")
                .dispatchIndirect(m_resources.indirectArguments)
                .IC_GI_USER_DATA
                .read(m_resources.priorityQueue, ResourceUsage::StorageBuffer)
                .write(m_resources.surfels, ResourceUsage::StorageBuffer)
                .write(m_resources.visibilityMoments, ResourceUsage::StorageBuffer)
                .write(m_resources.counters, ResourceUsage::StorageBuffer)
                .write(m_resources.diagnostics, ResourceUsage::StorageBuffer)
                .write(m_resources.hitRecords, ResourceUsage::StorageBuffer)
                .read(m_resources.probes, ResourceUsage::StorageBuffer)
                .read(m_resources.probeStaging, ResourceUsage::StorageBuffer)
                .asyncEligible();
            if (rayTracingSceneToken != InvalidGraphResourceId)
                builder.read(trace, rayTracingSceneToken,
                    ResourceUsage::StorageBuffer);
            m_passes.traceSurfelUpdates = trace;
        }

        auto preserveProbes = builder.addComputePass("GI.PreserveProbes", queue)
            .pipeline("gi_preserve_probes")
            .dispatch(static_cast<uint32_t>((probeCount + 63u) / 64u), 1u, 1u)
            .IC_GI_USER_DATA
            .write(m_resources.probes, ResourceUsage::StorageBuffer)
            .write(m_resources.probeStaging, ResourceUsage::StorageBuffer)
            .asyncEligible();
        m_passes.preserveProbes = preserveProbes;

        auto updateProbes = builder.addComputePass("GI.UpdateProbes", queue)
            .pipeline("gi_update_probes")
            .dispatch(std::max(1u,
                (limits.maxProbeUpdates + 63u) / 64u), 1u, 1u)
            .IC_GI_USER_DATA
            .read(m_resources.surfels, ResourceUsage::StorageBuffer)
            .read(m_resources.visibilityMoments, ResourceUsage::StorageBuffer)
            .read(m_resources.hashBuckets, ResourceUsage::StorageBuffer)
            .write(m_resources.probes, ResourceUsage::StorageBuffer)
            .write(m_resources.probeStaging, ResourceUsage::StorageBuffer)
            .write(m_resources.diagnostics, ResourceUsage::StorageBuffer)
            .write(m_resources.residualInterface, ResourceUsage::StorageBuffer)
            .asyncEligible();
        if (rayTracingSceneToken != InvalidGraphResourceId)
            builder.read(updateProbes, rayTracingSceneToken,
                ResourceUsage::StorageBuffer);
        m_passes.updateProbes = updateProbes;

        auto evaluate = builder.addComputePass("GI.EvaluateDiffuse", queue)
            .pipeline("gi_evaluate")
            .dispatch((width + 7u) / 8u, (height + 7u) / 8u, 1)
            .IC_GI_USER_DATA
            .read(m_resources.surfaceData, ResourceUsage::StorageBuffer)
            .read(m_resources.surfels, ResourceUsage::StorageBuffer)
            .read(m_resources.visibilityMoments, ResourceUsage::StorageBuffer)
            .read(m_resources.hashBuckets, ResourceUsage::StorageBuffer)
            .read(m_resources.probes, ResourceUsage::StorageBuffer)
            .read(m_resources.probeStaging, ResourceUsage::StorageBuffer)
            .read(m_resources.hitRecords, ResourceUsage::StorageBuffer)
            .write(m_resources.rawIrradiance, ResourceUsage::StorageTexture)
            .write(m_resources.diagnostics, ResourceUsage::StorageBuffer)
            .asyncEligible();
        m_passes.evaluate = evaluate;

        auto temporal = builder.addComputePass("GI.TemporalReconstruction", queue)
            .pipeline("gi_temporal_reconstruct")
            .dispatch((renderExtent.width + 7u) / 8u,
                (renderExtent.height + 7u) / 8u, 1)
            .IC_GI_USER_DATA
            .read(m_resources.rawIrradiance, ResourceUsage::SampledTexture)
            .read(sceneDepth, ResourceUsage::SampledTexture)
            .read(surfaceAttributes, ResourceUsage::SampledTexture)
            .write(m_resources.resolvedIrradiance, ResourceUsage::StorageTexture)
            .write(m_resources.diagnostics, ResourceUsage::StorageBuffer)
            .asyncEligible();
        builder.readPrevious(
            temporal, m_resources.resolvedIrradiance,
            ResourceUsage::SampledTexture);
        builder.readPrevious(
            temporal, sceneDepth, ResourceUsage::SampledTexture);
        builder.readPrevious(
            temporal, surfaceAttributes, ResourceUsage::SampledTexture);
        m_passes.temporalReconstruction = temporal;
        if (m_resources.diagnosticsReadback != InvalidGraphResourceId)
        {
            builder.addTransferPass("GI.ReadbackDiagnostics")
                .copy(m_resources.diagnostics,
                    m_resources.diagnosticsReadback);
        }
        return { .resolvedDiffuseIrradiance =
            m_resources.resolvedIrradiance };
#undef IC_GI_USER_DATA
    }
}
