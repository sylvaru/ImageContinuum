#include "common/tests_pch.h"
#include "global_illumination_tests.h"

#include "ic/renderer/frame_graph/frame_graph_arena.h"
#include "ic/renderer/frame_graph/frame_graph_builder.h"
#include "ic/renderer/frame_graph/frame_graph_compiler.h"
#include "ic/renderer/global_illumination/global_illumination.h"
#include "ic/renderer/global_illumination/global_illumination_bindings.h"

#include <spdlog/spdlog.h>

namespace
{
    int g_failures = 0;

    void check(bool condition, const char* message)
    {
        if (!condition)
        {
            ++g_failures;
            spdlog::error("[GlobalIlluminationTests] FAIL: {}", message);
        }
    }

    ic::GlobalIlluminationCapabilities supportedCapabilities()
    {
        ic::GlobalIlluminationCapabilities caps{};
        caps.rayTracing.accelerationStructures = true;
        caps.rayTracing.inlineRayQueries = true;
        caps.rayTracing.sharedAccelerationStructures = true;
        caps.rayTracing.indirectDispatch = true;
        caps.storageImages = true;
        caps.indirectDispatch = true;
        caps.asyncCompute = true;
        return caps;
    }

    const ic::GraphResource* resource(
        const ic::FrameGraphBuilder& builder, ic::GraphResourceId id)
    {
        return id < builder.resources().size() ? &builder.resources()[id] : nullptr;
    }
}

int runGlobalIlluminationTests()
{
    using namespace ic;
    spdlog::warn("[GlobalIlluminationTests] running...");
    g_failures = 0;

    {
        const GlobalIlluminationLimits clamped =
            GlobalIlluminationSystem::sanitizeLimits({
                .maxSurfels = UINT32_MAX,
                .hashBucketCount = 0,
                .maxSurfelUpdates = UINT32_MAX,
                .probeClipmapCount = UINT32_MAX,
                .probeResolution = UINT32_MAX,
                .maxProbeUpdates = 0,
                .rayBudget = UINT32_MAX });
        check(clamped.maxSurfels == 4u * 1024u * 1024u,
            "surfel capacity is overflow-safe");
        check(clamped.hashBucketCount == 1u,
            "hash capacity has a non-zero floor");
        check(clamped.probeClipmapCount == 8u &&
              clamped.probeResolution == 64u,
            "probe volume dimensions are bounded");
    }

    {
        const auto views = globalIlluminationDebugViews();
        check(views.size() == static_cast<size_t>(
                  GlobalIlluminationDebugView::Count),
            "every GI debug enum has a metadata entry");
        for (size_t i = 0; i < views.size(); ++i)
        {
            check(static_cast<size_t>(views[i].view) == i &&
                  !views[i].name.empty() && !views[i].configName.empty() &&
                  !views[i].tooltip.empty() && !views[i].legend.empty() &&
                  !views[i].unitsAndRange.empty() &&
                  !views[i].sourcePass.empty() &&
                  !views[i].sourceResource.empty(),
                "GI debug metadata is complete and index-stable");
            for (size_t j = 0; j < i; ++j)
                check(views[i].configName != views[j].configName,
                    "GI debug configuration names are unique");
        }

        GlobalIlluminationConfiguration base{};
        base.diagnosticIntensity = 2.25f;
        base.debugExposure = 0.5f;
        const std::array<GlobalIlluminationQuality, 5> qualities = {
            GlobalIlluminationQuality::Off,
            GlobalIlluminationQuality::Low,
            GlobalIlluminationQuality::Medium,
            GlobalIlluminationQuality::High,
            GlobalIlluminationQuality::Ultra };
        const std::array<uint32_t, 5> expectedResolution = { 16, 12, 16, 24, 32 };
        const std::array<uint32_t, 5> expectedUpdates = { 512, 256, 512, 1024, 2048 };
        const std::array<uint32_t, 5> expectedRays = { 12288, 4096, 12288, 32768, 65536 };
        for (size_t i = 0; i < qualities.size(); ++i)
        {
            const auto preset = globalIlluminationPreset(qualities[i], base);
            check(preset.limits.probeResolution == expectedResolution[i] &&
                  preset.limits.maxProbeUpdates == expectedUpdates[i] &&
                  preset.limits.rayBudget == expectedRays[i],
                "GI preset budgets are exact and coherent");
            check(preset.diagnosticIntensity == base.diagnosticIntensity &&
                  preset.debugExposure == base.debugExposure &&
                  !preset.multiBounce && preset.probeFallback,
                "GI presets preserve display/energy controls and avoid unfinished feedback");
            const uint64_t bytes = GlobalIlluminationSystem::estimateAllocatedBytes(
                preset, {1920, 1080});
            if (qualities[i] == GlobalIlluminationQuality::Off)
                check(!preset.enabled && bytes == 0,
                    "Off allocates no logical GI resources");
            else
                check(preset.enabled && bytes <=
                      globalIlluminationPresetInfo(qualities[i]).memoryLimitBytes,
                    "enabled preset fits its declared 1080p memory limit");
        }
    }

    FrameGraphArena arena;
    FrameGraphBuilder builder(arena.resource());
    const GraphResourceId depth = builder.createHistoryTexture({
        .width = 1280, .height = 720, .format = TextureFormat::D32_Float,
        .usage = TextureUsageFlags::Sampled |
            TextureUsageFlags::DepthAttachment,
        .debugName = "Test.Depth" });
    const GraphResourceId attributes = builder.createHistoryTexture({
        .width = 1280, .height = 720,
        .format = TextureFormat::RGBA32_Float,
        .usage = TextureUsageFlags::Sampled |
            TextureUsageFlags::ColorAttachment,
        .debugName = "Test.SurfaceAttributes" });

    GlobalIlluminationSystem gi;
    GlobalIlluminationConfiguration config{};
    config.enabled = true;
    gi.configure(config);

    check(!gi.contributePasses(builder, {
            .sceneDepth = depth,
            .surfaceAttributes = attributes,
            .renderExtent = {1280, 720},
            .asyncComputeEnabled = true }).valid(),
        "unsupported hardware schedules no GI passes or resources");
    check(builder.nodes().empty(),
        "unsupported fallback preserves the original graph");

    gi.setCapabilities(supportedCapabilities());
    // Exercise the optional surfel-detail path below; probe-primary DDGI is
    // covered independently after the full dependency graph assertions.
    config.surfelDetail = true;
    gi.configure(config);
    const GraphResourceId rtToken = builder.createPersistentBuffer({
        .size = 16, .usage = BufferUsageFlags::Storage,
        .debugName = "Test.RayTracingSceneToken" });
    const GraphResourceId resolved = gi.contributePasses(builder, {
        .sceneDepth = depth,
        .surfaceAttributes = attributes,
        .rayTracingSceneToken = rtToken,
        .renderExtent = {1280, 720},
        .asyncComputeEnabled = true }).resolvedDiffuseIrradiance;
    check(resolved != InvalidGraphResourceId,
        "supported GI declares a resolved irradiance output");
    check(builder.nodes().size() == 9u,
        "all nine hierarchical GI passes are declared");

    const GraphNodeId traceNode = gi.passes().traceSurfelUpdates;
    const uint32_t tracePayloadIndex =
        builder.nodes()[traceNode].graphNode.payloadIndex;
    const ComputePassData* tracePayload = std::get_if<ComputePassData>(
        &builder.payloads()[tracePayloadIndex]);
    check(tracePayload &&
          tracePayload->indirectArguments == gi.resources().indirectArguments,
        "surfel RT updates use graph-owned indirect dispatch arguments");
    const uint32_t expectedUpdateBits =
        (config.limits.maxSurfelUpdates & 0x3ffffu) |
        (((config.limits.maxProbeUpdates / 64u) & 0xfffu) << 18u);
    check(tracePayload && tracePayload->userData[0] == expectedUpdateBits &&
          tracePayload->userData[1] == config.limits.rayBudget &&
          tracePayload->userData[2] == 16u,
        "ray budget and deterministic rays-per-surfel cross the generic pass payload");

    std::array<bool, GiBufferBindingCount> seenBindings{};
    for (uint32_t i = 0; i < GiBufferBindingCount; ++i)
    {
        check(GiBufferSemantics[i] != GraphResourceSemantic::Generic,
            "GI binding schema has no unspecified backend slots");
        for (uint32_t j = 0; j < i; ++j)
            check(GiBufferSemantics[i] != GiBufferSemantics[j],
                "GI binding schema is one-to-one across Vulkan and DX12");
        seenBindings[i] = true;
    }
    check(std::ranges::all_of(seenBindings, [](bool value) { return value; }),
        "all shared GI buffer bindings are covered");

    const auto& resources = gi.resources();
    const GraphResource* surfels = resource(builder, resources.surfels);
    const GraphResource* surface = resource(builder, resources.surfaceData);
    const GraphResource* history = resource(builder, resources.resolvedIrradiance);
    const GraphResource* hitRecords = resource(builder, resources.hitRecords);
    check(surfels && surfels->ownership == ResourceOwnership::Persistent &&
              surfels->multiplicity == ResourceMultiplicity::Single,
        "surfel cache is persistent single-instance state");
    check(surface && surface->ownership == ResourceOwnership::Transient &&
              surface->multiplicity == ResourceMultiplicity::PerFrameSlot,
        "surface staging is isolated per frame slot");
    check(history && history->multiplicity == ResourceMultiplicity::History,
        "resolved irradiance uses an explicit history ring");
    check(hitRecords &&
          hitRecords->ownership == ResourceOwnership::Persistent &&
          hitRecords->multiplicity == ResourceMultiplicity::Single &&
          hitRecords->bufferDesc.size == static_cast<uint64_t>(
              config.limits.maxSurfels) * sizeof(GpuGiHitRecord),
        "stable surfel hit records persist without per-frame allocation");

    FrameGraphCompiler compiler(arena.resource());
    const CompiledGraphPlan plan = compiler.compile(builder);
    check(plan.executionOrder.size() == builder.nodes().size(),
        "GI dependency graph compiles without cycles");

    {
        FrameGraphArena probeArena;
        FrameGraphBuilder probeBuilder(probeArena.resource());
        const GraphResourceId probeDepth = probeBuilder.createHistoryTexture({
            .width = 64, .height = 64, .format = TextureFormat::D32_Float,
            .usage = TextureUsageFlags::Sampled |
                TextureUsageFlags::DepthAttachment,
            .debugName = "ProbePrimary.Depth" });
        const GraphResourceId probeAttributes =
            probeBuilder.createHistoryTexture({
                .width = 64, .height = 64,
                .format = TextureFormat::RGBA32_Float,
                .usage = TextureUsageFlags::Sampled |
                    TextureUsageFlags::ColorAttachment,
                .debugName = "ProbePrimary.SurfaceAttributes" });
        const GraphResourceId probeRtToken =
            probeBuilder.createPersistentBuffer({
                .size = 16, .usage = BufferUsageFlags::Storage,
                .debugName = "ProbePrimary.RayTracingSceneToken" });
        GlobalIlluminationSystem probeGi;
        probeGi.setCapabilities(supportedCapabilities());
        GlobalIlluminationConfiguration probeConfig{};
        probeConfig.enabled = true;
        probeConfig.surfelDetail = false;
        probeGi.configure(probeConfig);
        check(probeGi.contributePasses(probeBuilder, {
                  .sceneDepth = probeDepth,
                  .surfaceAttributes = probeAttributes,
                  .rayTracingSceneToken = probeRtToken,
                  .renderExtent = {64, 64},
                  .asyncComputeEnabled = true }).valid() &&
              probeBuilder.nodes().size() == 6u &&
              probeGi.passes().traceSurfelUpdates == InvalidGraphNodeId,
            "probe-primary DDGI omits all surfel update passes");
        const GraphResource* probeSurfels = resource(
            probeBuilder, probeGi.resources().surfels);
        const GraphResource* completedProbes = resource(
            probeBuilder, probeGi.resources().probes);
        const GraphResource* stagingProbes = resource(
            probeBuilder, probeGi.resources().probeStaging);
        check(probeSurfels && probeSurfels->bufferDesc.size ==
                  sizeof(GpuRadianceSurfel),
            "probe-primary DDGI retains only a binding-safe surfel sentinel");
        check(completedProbes && stagingProbes &&
              completedProbes->ownership == ResourceOwnership::Persistent &&
              stagingProbes->ownership == ResourceOwnership::Persistent &&
              completedProbes->multiplicity == ResourceMultiplicity::Single &&
              stagingProbes->multiplicity == ResourceMultiplicity::Single &&
              completedProbes->bufferDesc.size == stagingProbes->bufferDesc.size,
            "probe-primary DDGI owns two equal persistent snapshot fields");
        FrameGraphCompiler probeCompiler(probeArena.resource());
        const CompiledGraphPlan probePlan = probeCompiler.compile(probeBuilder);
        bool probePreserveWaitsForPriorReaders = false;
        bool stagingPreserveWaitsForPriorReaders = false;
        bool updateWritesCompleted = false;
        bool updateWritesStaging = false;
        bool preserveWritesCompleted = false;
        bool preserveWritesStaging = false;
        bool evaluateReadsCompleted = false;
        bool evaluateReadsStaging = false;
        for (const CrossFrameDependency& dependency :
             probePlan.crossFrameDependencies)
        {
            probePreserveWaitsForPriorReaders |=
                dependency.resource == probeGi.resources().probes &&
                dependency.consumerNode == probeGi.passes().preserveProbes;
            stagingPreserveWaitsForPriorReaders |=
                dependency.resource == probeGi.resources().probeStaging &&
                dependency.consumerNode == probeGi.passes().preserveProbes;
        }
        for (const ResourceAccess& access : probePlan.resourceAccesses)
        {
            if (access.node == probeGi.passes().updateProbes &&
                access.access == AccessType::Write)
            {
                updateWritesCompleted |=
                    access.resource == probeGi.resources().probes;
                updateWritesStaging |=
                    access.resource == probeGi.resources().probeStaging;
            }
            if (access.node == probeGi.passes().preserveProbes &&
                access.access == AccessType::Write)
            {
                preserveWritesCompleted |=
                    access.resource == probeGi.resources().probes;
                preserveWritesStaging |=
                    access.resource == probeGi.resources().probeStaging;
            }
            if (access.node == probeGi.passes().evaluate &&
                access.access == AccessType::Read)
            {
                evaluateReadsCompleted |=
                    access.resource == probeGi.resources().probes;
                evaluateReadsStaging |=
                    access.resource == probeGi.resources().probeStaging;
            }
        }
        check(probePreserveWaitsForPriorReaders,
            "per-frame probe preservation waits for the prior frame's last read");
        check(stagingPreserveWaitsForPriorReaders,
            "per-frame staging preservation waits for the prior frame's last read");
        check(updateWritesCompleted && updateWritesStaging,
            "probe update declares both parity-selected snapshot fields");
        check(preserveWritesCompleted && preserveWritesStaging,
            "probe preservation declares both parity-selected snapshot fields");
        check(evaluateReadsCompleted && evaluateReadsStaging,
            "probe evaluation declares both parity-selected snapshot fields");
    }

    bool readsPreviousResolved = false;
    bool opaqueReadyDependency = false;
    bool traceReadsRayTracingScene = false;
    bool traceWritesHitRecords = false;
    for (const ResourceAccess& access : plan.resourceAccesses)
    {
        readsPreviousResolved |= access.resource == resolved &&
            access.previousVersion;
        traceReadsRayTracingScene |= access.node == traceNode &&
            access.resource == rtToken && access.access == AccessType::Read;
        traceWritesHitRecords |= access.node == traceNode &&
            access.resource == resources.hitRecords &&
            access.access == AccessType::Write;
    }
    for (const Dependency& dependency : plan.dependencies)
    {
        opaqueReadyDependency |=
            dependency.source == gi.passes().evaluate &&
            dependency.destination == gi.passes().temporalReconstruction;
    }
    check(readsPreviousResolved,
        "temporal reconstruction reads the previous history instance");
    check(opaqueReadyDependency,
        "evaluation orders temporal reconstruction through raw irradiance");
    check(traceReadsRayTracingScene,
        "GI trace pass waits on the shared TLAS build token");
    check(traceWritesHitRecords,
        "GI trace pass owns the shared hit-record write dependency");
    check(GiIndexBuffersBinding ==
              GiVertexBuffersBinding + GiMaxGeometryBufferCount &&
          GiBindlessTexturesBinding ==
              GiIndexBuffersBinding + GiMaxGeometryBufferCount,
        "DX12 descriptor ranges are non-overlapping and parity-bounded");

    const auto before = gi.statistics().historyEpoch;
    gi.invalidateHistory();
    check(gi.statistics().historyEpoch == before + 1u &&
          !gi.statistics().historyValid,
        "history invalidation advances a stable epoch");

    FrameGraphArena disabledArena;
    FrameGraphBuilder disabledBuilder(disabledArena.resource());
    const GraphResourceId disabledDepth = disabledBuilder.createHistoryTexture({
        .width = 64, .height = 64, .format = TextureFormat::D32_Float,
        .usage = TextureUsageFlags::Sampled |
            TextureUsageFlags::DepthAttachment,
        .debugName = "Disabled.Depth" });
    const GraphResourceId disabledAttributes =
        disabledBuilder.createHistoryTexture({
            .width = 64, .height = 64,
            .format = TextureFormat::RGBA32_Float,
            .usage = TextureUsageFlags::Sampled |
                TextureUsageFlags::ColorAttachment,
            .debugName = "Disabled.SurfaceAttributes" });
    config.enabled = false;
    gi.configure(config);
    check(!gi.contributePasses(disabledBuilder, {
              .sceneDepth = disabledDepth,
              .surfaceAttributes = disabledAttributes,
              .renderExtent = {64, 64},
              .asyncComputeEnabled = true }).valid() &&
              disabledBuilder.nodes().empty(),
        "runtime disable removes GI scheduling deterministically");

    if (g_failures == 0)
        spdlog::warn("[GlobalIlluminationTests] PASSED");
    return g_failures;
}
