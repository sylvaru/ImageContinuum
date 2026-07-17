#include "common/tests_pch.h"
#include "frame_graph_resource_tests.h"

#include "ic/renderer/frame_graph/frame_graph_types.h"
#include "ic/renderer/frame_graph/frame_graph_builder.h"
#include "ic/renderer/frame_graph/frame_graph_compiler.h"
#include "ic/renderer/gpu_driven_submission.h"
#include "ic/renderer/renderer_common/renderer_util.h"

#include <spdlog/spdlog.h>

#include <memory_resource>
#include <set>

using namespace ic;

namespace
{
    int g_failures = 0;

    void check(bool condition, const char* what)
    {
        if (!condition)
        {
            ++g_failures;
            spdlog::error("[FrameGraphResourceTests] FAIL: {}", what);
        }
    }

    // ---- Property 1: instance-index selection math -----------------------
    //
    // Covers: Single always resolves to instance 0; PerFrameSlot/History
    // resolve to distinct instances across frame slots; History current/
    // previous wrap correctly for 2 and 3 frames in flight; previous(counter)
    // == current(counter - 1) (so advancement by exactly one submitted frame
    // always points "previous" at the immediately preceding submitted frame).
    void testInstanceIndexMath()
    {
        // Single: one instance regardless of frame counter.
        for (uint64_t counter = 0; counter < 8; ++counter)
        {
            check(currentFrameInstanceIndex(counter, 1) == 0,
                "Single current index is always 0");
            check(previousFrameInstanceIndex(counter, 1) == 0,
                "Single previous index is always 0");
        }

        // Per-frame-slot / history rings for 2 and 3 frames in flight.
        for (uint32_t n : {2u, 3u})
        {
            std::set<uint32_t> seenCurrent;
            for (uint64_t counter = 0; counter < 4ull * n; ++counter)
            {
                const uint32_t cur = currentFrameInstanceIndex(counter, n);
                const uint32_t prev = previousFrameInstanceIndex(counter, n);

                check(cur < n, "current index in range");
                check(prev < n, "previous index in range");
                check(cur != prev,
                    "current and previous are distinct instances");

                if (counter > 0)
                {
                    check(prev == currentFrameInstanceIndex(counter - 1, n),
                        "previous == prior submitted frame's current");
                }

                if (counter < n)
                {
                    seenCurrent.insert(cur);
                }
            }
            // Over one full period the ring visits every physical instance.
            check(seenCurrent.size() == n,
                "per-frame-slot ring covers every instance");
        }
    }

    bool hasDependencyEdge(
        const CompiledGraphPlan& plan, GraphNodeId a, GraphNodeId b)
    {
        for (const Dependency& d : plan.dependencies)
        {
            if ((d.source == a && d.destination == b) ||
                (d.source == b && d.destination == a))
            {
                return true;
            }
        }
        return false;
    }

    // ---- Property 2: compiler treats versions as distinct instances ------
    //
    // A pass that reads the PREVIOUS instance of a history resource must NOT
    // gain an intra-frame dependency on the pass that writes the CURRENT
    // instance of the same resource (that would be a false dependency, and if
    // symmetric, a cycle). The transition of the previous instance must still
    // be emitted, tagged previousVersion so the backend targets previousEntry.
    void testHistoryHasNoFalseDependency()
    {
        std::pmr::monotonic_buffer_resource mem;
        FrameGraphBuilder builder(&mem);

        const GraphResourceId history = builder.createHistoryTexture({
            .width = 64, .height = 64,
            .format = TextureFormat::RGBA8_UNorm,
            .usage = TextureUsageFlags::Sampled | TextureUsageFlags::Storage,
            .debugName = "Test.History" });

        const GraphNodeId producer = builder.addComputePass("WriteCurrent");
        builder.write(producer, history, ResourceUsage::StorageTexture);

        const GraphNodeId consumer = builder.addComputePass("ReadPrevious");
        builder.readPrevious(consumer, history, ResourceUsage::SampledTexture);

        FrameGraphCompiler compiler(&mem);
        const CompiledGraphPlan plan = compiler.compile(builder);

        check(!hasDependencyEdge(plan, producer, consumer),
            "read-previous does not depend on this frame's write-current");
        check(plan.executionOrder.size() == plan.nodes.size(),
            "graph is acyclic and fully scheduled with a history cycle");

        bool taggedPreviousBarrier = false;
        bool currentBarrierForHistory = false;
        for (const ResourceBarrier& b : plan.barriers)
        {
            if (b.resource != history)
            {
                continue;
            }
            if (b.previousVersion)
            {
                taggedPreviousBarrier = true;
            }
            else
            {
                currentBarrierForHistory = true;
            }
        }
        check(taggedPreviousBarrier,
            "previous-instance read emits a previousVersion barrier");
        // The current instance is only written (single access) so it needs no
        // extra transition; the important property is the previous barrier is
        // separate and tagged, not mixed with the current version.
        (void)currentBarrierForHistory;
    }

    // Positive control: reading the CURRENT version of an ordinary resource
    // DOES create a real producer -> consumer dependency, proving the absence
    // of a dependency above is specific to previous-version reads.
    void testCurrentReadCreatesDependency()
    {
        std::pmr::monotonic_buffer_resource mem;
        FrameGraphBuilder builder(&mem);

        const GraphResourceId buffer = builder.createBuffer({
            .size = 256,
            .usage = BufferUsageFlags::Storage,
            .debugName = "Test.Current" });

        const GraphNodeId producer = builder.addComputePass("Write");
        builder.write(producer, buffer, ResourceUsage::StorageBuffer);

        const GraphNodeId consumer = builder.addComputePass("Read");
        builder.read(consumer, buffer, ResourceUsage::StorageBuffer);

        FrameGraphCompiler compiler(&mem);
        const CompiledGraphPlan plan = compiler.compile(builder);

        check(hasDependencyEdge(plan, producer, consumer),
            "current-version read depends on the current-version write");
    }

    // A history resource written this frame AND read-previous by another pass,
    // plus a normal current read of a different resource, must still compile to
    // an acyclic schedule (no cycle introduced by mixing versions).
    void testMixedVersionGraphIsAcyclic()
    {
        std::pmr::monotonic_buffer_resource mem;
        FrameGraphBuilder builder(&mem);

        const GraphResourceId history = builder.createHistoryBuffer({
            .size = 512,
            .usage = BufferUsageFlags::Storage,
            .debugName = "Test.HistoryBuffer" });
        const GraphResourceId scratch = builder.createBuffer({
            .size = 512,
            .usage = BufferUsageFlags::Storage,
            .debugName = "Test.Scratch" });

        const GraphNodeId a = builder.addComputePass("A");
        builder.write(a, scratch, ResourceUsage::StorageBuffer);
        builder.write(a, history, ResourceUsage::StorageBuffer);

        const GraphNodeId b = builder.addComputePass("B");
        builder.read(b, scratch, ResourceUsage::StorageBuffer);
        builder.readPrevious(b, history, ResourceUsage::StorageBuffer);

        FrameGraphCompiler compiler(&mem);
        const CompiledGraphPlan plan = compiler.compile(builder);

        // B depends on A only through scratch (current), not through history
        // (previous). The schedule must be acyclic and complete.
        check(hasDependencyEdge(plan, a, b),
            "mixed graph keeps the current-resource dependency");
        check(plan.executionOrder.size() == plan.nodes.size(),
            "mixed-version graph is acyclic and fully scheduled");
    }

    // ---- Item 2: GPU-driven buffers are graph-owned (no dual model) --------
    //
    // Each GPU-driven buffer is declared exactly once in the graph (a single
    // allocation, not a graph "token" duplicated by a backend buffer), inputs
    // are CPU-uploadable and outputs GPU-only, and the indirect-argument buffer
    // is declared by element count so each backend sizes it at its native
    // command stride.
    int countSemantic(
        const CompiledGraphPlan& plan, GraphResourceSemantic semantic)
    {
        int n = 0;
        for (const GraphResource& r : plan.resources)
        {
            if (r.semantic == semantic) ++n;
        }
        return n;
    }

    const CrossFrameDependency* findCrossFrame(
        const CompiledGraphPlan& plan, GraphResourceId resource);

    void testGpuDrivenBuffersAreGraphOwned()
    {
        std::pmr::monotonic_buffer_resource mem;
        FrameGraphBuilder builder(&mem);

        const GpuDrivenGraphResources res =
            declareGpuDrivenResources(builder);
        addGpuDrivenCullPasses(builder, res);

        FrameGraphCompiler compiler(&mem);
        const CompiledGraphPlan plan = compiler.compile(builder);

        const GraphResourceSemantic semantics[] = {
            GraphResourceSemantic::GpuDrivenInstanceBounds,
            GraphResourceSemantic::GpuDrivenDrawInputs,
            GraphResourceSemantic::GpuDrivenVisibleInstances,
            GraphResourceSemantic::GpuDrivenVisibleCount,
            GraphResourceSemantic::GpuDrivenIndirectArguments,
            GraphResourceSemantic::GpuDrivenDrawMetadata,
            GraphResourceSemantic::GpuDrivenBinCounts };
        for (GraphResourceSemantic s : semantics)
        {
            check(countSemantic(plan, s) == 1,
                "each GPU-driven buffer has exactly one graph allocation");
            check(findResourceBySemantic(plan, s) != InvalidGraphResourceId,
                "each GPU-driven buffer resolves by semantic");
        }

        // Indirect args: element-count declared, byte size deferred to backend.
        const GraphResourceId indirect = findResourceBySemantic(
            plan, GraphResourceSemantic::GpuDrivenIndirectArguments);
        check(plan.resources[indirect].bufferDesc.elementCount ==
                  MaxGpuDrivenDraws,
            "indirect args declared by element count");
        check(plan.resources[indirect].bufferDesc.size == 0,
            "indirect args byte size resolved per backend (0 in the graph)");

        // Inputs are CPU-uploadable; outputs are GPU-only.
        const GraphResourceId bounds = findResourceBySemantic(
            plan, GraphResourceSemantic::GpuDrivenInstanceBounds);
        const GraphResourceId inputs = findResourceBySemantic(
            plan, GraphResourceSemantic::GpuDrivenDrawInputs);
        const GraphResourceId visible = findResourceBySemantic(
            plan, GraphResourceSemantic::GpuDrivenVisibleInstances);
        check(plan.resources[bounds].bufferDesc.memoryUsage ==
                  ResourceMemoryUsage::CpuToGpu,
            "instance bounds are CPU-uploadable");
        check(plan.resources[inputs].bufferDesc.memoryUsage ==
                  ResourceMemoryUsage::CpuToGpu,
            "draw inputs are CPU-uploadable");
        check(plan.resources[visible].bufferDesc.memoryUsage ==
                  ResourceMemoryUsage::GpuOnly,
            "cull outputs are GPU-only");

        // All GPU-driven buffers are per-frame-slot instanced (item 1 model).
        check(plan.resources[indirect].multiplicity ==
                  ResourceMultiplicity::PerFrameSlot,
            "GPU-driven buffers are per-frame-slot instanced");

        // GPU-driven per-frame-slot buffers never force cross-frame
        // serialization: consecutive frames use distinct ring instances.
        check(plan.crossFrameDependencies.empty(),
            "per-frame-slot GPU-driven graph creates no cross-frame dependency");
    }

    void testGpuCullReadsPreviousHiZ()
    {
        std::pmr::monotonic_buffer_resource mem;
        FrameGraphBuilder builder(&mem);

        const GraphResourceId hiZ = builder.createHistoryTexture({
            .width = 128,
            .height = 64,
            .mipLevels = 8,
            .format = TextureFormat::R32_Float,
            .usage =
                TextureUsageFlags::Sampled |
                TextureUsageFlags::Storage,
            .debugName = "Test.GpuCullHiZ" });
        const GpuDrivenGraphResources resources =
            declareGpuDrivenResources(builder);
        const GraphNodeId cull =
            addGpuDrivenCullPasses(builder, resources, hiZ);
        const GraphNodeId build = builder.addComputePass("BuildCurrentHiZ");
        builder.write(build, hiZ, ResourceUsage::StorageTexture);

        FrameGraphCompiler compiler(&mem);
        const CompiledGraphPlan plan = compiler.compile(builder);

        check(plan.resources[hiZ].multiplicity ==
                  ResourceMultiplicity::History,
            "GPU occlusion pyramid is a history resource");
        check(findPreviousNodeResource(
                  plan, plan.nodes[cull], ResourceUsage::SampledTexture) ==
                  hiZ,
            "GPU cull explicitly reads the previous Hi-Z instance");
        check(!hasDependencyEdge(plan, build, cull),
            "previous Hi-Z read does not depend on the current-frame build");

        const CrossFrameDependency* dep = findCrossFrame(plan, hiZ);
        check(dep != nullptr,
            "previous Hi-Z cull emits a precise cross-frame edge");
        if (dep)
        {
            check(dep->consumerNode == cull,
                "previous Hi-Z cross-frame consumer is the cull pass");
        }
    }

    void testGpuCullDiagnosticsAreExplicitAndOptional()
    {
        {
            std::pmr::monotonic_buffer_resource mem;
            FrameGraphBuilder builder(&mem);
            const GpuDrivenGraphResources resources =
                declareGpuDrivenResources(builder, false);

            check(resources.cullStatsReadback == InvalidGraphResourceId,
                "disabled diagnostics allocate no readback buffer");
            check(builder.resources()[resources.cullClassification]
                      .bufferDesc.size == sizeof(uint32_t),
                "disabled diagnostics keep only a bound classification placeholder");
        }

        std::pmr::monotonic_buffer_resource mem;
        FrameGraphBuilder builder(&mem);
        const GraphResourceId hiZ = builder.createHistoryTexture({
            .width = 128,
            .height = 64,
            .mipLevels = 8,
            .format = TextureFormat::R32_Float,
            .usage =
                TextureUsageFlags::Sampled |
                TextureUsageFlags::Storage,
            .debugName = "Test.DiagnosticsHiZ" });
        const GpuDrivenGraphResources resources =
            declareGpuDrivenResources(builder, true);
        addGpuDrivenCullPasses(builder, resources, hiZ);
        builder.addComputePass("BuildCurrentHiZ")
            .write(hiZ, ResourceUsage::StorageTexture);
        addGpuDrivenOcclusionValidationPasses(
            builder, resources, hiZ, QueueType::Compute);

        FrameGraphCompiler compiler(&mem);
        const CompiledGraphPlan plan = compiler.compile(builder);
        const GraphResourceId readback = findResourceBySemantic(
            plan, GraphResourceSemantic::GpuDrivenCullStatsReadback);
        check(readback != InvalidGraphResourceId,
            "enabled diagnostics declare a graph-owned stats readback");
        if (readback != InvalidGraphResourceId)
        {
            check(plan.resources[readback].bufferDesc.memoryUsage ==
                      ResourceMemoryUsage::GpuToCpu,
                "diagnostic readback uses GPU-to-CPU memory");
            check(plan.resources[readback].multiplicity ==
                      ResourceMultiplicity::PerFrameSlot,
                "diagnostic readback follows frame-slot lifetime rules");
        }

        bool foundValidation = false;
        bool foundReadback = false;
        for (const ExecutionNode& node : plan.nodes)
        {
            const auto& payload = plan.payloads[node.payloadIndex];
            if (const auto* compute = std::get_if<ComputePassData>(&payload))
            {
                foundValidation |=
                    compute->name == "ValidateGpuOcclusion";
            }
            else if (const auto* transfer =
                         std::get_if<TransferPassData>(&payload))
            {
                foundReadback |=
                    transfer->name == "ReadbackGpuCullStats";
            }
        }
        check(foundValidation,
            "enabled diagnostics add the focused validation pass");
        check(foundReadback,
            "enabled diagnostics add the focused readback pass");
    }

    void testOcclusionHistoryReliabilityFallbacks()
    {
        GpuOcclusionHistoryState history{};
        SceneRenderView scene{};
        scene.sceneVersion = 7;
        scene.camera.valid = 1;
        scene.camera.position = glm::vec3(0.0f, 1.0f, 5.0f);
        scene.camera.nearPlane = 0.1f;
        scene.camera.farPlane = 1000.0f;
        scene.camera.verticalFovRadians = glm::radians(70.0f);
        scene.camera.aspectRatio = 16.0f / 9.0f;
        scene.camera.view = glm::lookAtRH(
            scene.camera.position,
            glm::vec3(0.0f, 1.0f, 0.0f),
            glm::vec3(0.0f, 1.0f, 0.0f));

        auto projection = [&]()
        {
            return glm::perspectiveRH_ZO(
                scene.camera.verticalFovRadians,
                scene.camera.aspectRatio,
                scene.camera.nearPlane,
                scene.camera.farPlane);
        };
        GpuFrameData frame{};

        check(!updateGpuOcclusionHistory(
                  history, scene, 1920, 1080,
                  scene.camera.view, projection(), frame),
            "first frame falls back to frustum-only culling");
        check(updateGpuOcclusionHistory(
                  history, scene, 1920, 1080,
                  scene.camera.view, projection(), frame),
            "stable history enables previous-Hi-Z occlusion");

        scene.camera.verticalFovRadians = glm::radians(75.0f);
        check(!updateGpuOcclusionHistory(
                  history, scene, 1920, 1080,
                  scene.camera.view, projection(), frame),
            "projection change invalidates occlusion history");
        check(updateGpuOcclusionHistory(
                  history, scene, 1920, 1080,
                  scene.camera.view, projection(), frame),
            "history recovers after one frame at the new projection");

        scene.camera.position.x += 2.0f;
        scene.camera.view = glm::lookAtRH(
            scene.camera.position,
            glm::vec3(0.0f, 1.0f, 0.0f),
            glm::vec3(0.0f, 1.0f, 0.0f));
        check(!updateGpuOcclusionHistory(
                  history, scene, 1920, 1080,
                  scene.camera.view, projection(), frame),
            "camera cut invalidates occlusion history");
        check(!updateGpuOcclusionHistory(
                  history, scene, 1280, 720,
                  scene.camera.view, projection(), frame),
            "extent change invalidates occlusion history");

        ++scene.sceneVersion;
        check(!updateGpuOcclusionHistory(
                  history, scene, 1280, 720,
                  scene.camera.view, projection(), frame),
            "scene change invalidates occlusion history");
    }

    // ---- Cross-frame dependency generation (multi-frame pipelining) --------
    //
    // The executor no longer serializes every frame behind the previous frame's
    // whole graph; it applies only the edges the compiler derives here. These
    // tests pin that derivation: per-frame-slot overlaps freely, a written
    // Single/persistent resource emits exactly one WAR/WAW edge, a read-only
    // persistent resource emits none, and a history previous-read emits a RAW
    // edge on the reader.

    const CrossFrameDependency* findCrossFrame(
        const CompiledGraphPlan& plan, GraphResourceId resource)
    {
        for (const CrossFrameDependency& dep : plan.crossFrameDependencies)
        {
            if (dep.resource == resource)
            {
                return &dep;
            }
        }
        return nullptr;
    }

    void testPerFrameSlotHasNoCrossFrameDependency()
    {
        std::pmr::monotonic_buffer_resource mem;
        FrameGraphBuilder builder(&mem);

        const GraphResourceId buffer = builder.createBuffer({
            .size = 256,
            .usage = BufferUsageFlags::Storage,
            .debugName = "Test.PerFrameSlot" });

        const GraphNodeId writer = builder.addComputePass("Write");
        builder.write(writer, buffer, ResourceUsage::StorageBuffer);
        const GraphNodeId reader = builder.addComputePass("Read");
        builder.read(reader, buffer, ResourceUsage::StorageBuffer);

        FrameGraphCompiler compiler(&mem);
        const CompiledGraphPlan plan = compiler.compile(builder);

        check(plan.crossFrameDependencies.empty(),
            "per-frame-slot written+read resource creates no cross-frame edge");
    }

    void testPersistentWriteEmitsCrossFrameEdge()
    {
        std::pmr::monotonic_buffer_resource mem;
        FrameGraphBuilder builder(&mem);

        const GraphResourceId persistent = builder.createPersistentBuffer({
            .size = 256,
            .usage = BufferUsageFlags::Storage,
            .debugName = "Test.Persistent" });

        const GraphNodeId writer = builder.addComputePass("Build");
        builder.write(writer, persistent, ResourceUsage::StorageBuffer);
        const GraphNodeId reader = builder.addComputePass("Consume");
        builder.read(reader, persistent, ResourceUsage::StorageBuffer);

        FrameGraphCompiler compiler(&mem);
        const CompiledGraphPlan plan = compiler.compile(builder);

        const CrossFrameDependency* dep = findCrossFrame(plan, persistent);
        check(dep != nullptr,
            "written persistent resource emits a cross-frame edge");
        check(plan.crossFrameDependencies.size() == 1,
            "exactly one cross-frame edge for a single written persistent");
        if (dep)
        {
            // The new writer is the consumer (it must wait for the prior frame's
            // last reader); the producer submission is that last reader's.
            check(dep->consumerNode == writer,
                "cross-frame consumer is the first writer");
            check(dep->consumerSubmission != dep->producerSubmission,
                "writer waits on the prior frame's later reader submission");
        }
    }

    void testReadOnlyPersistentHasNoCrossFrameEdge()
    {
        std::pmr::monotonic_buffer_resource mem;
        FrameGraphBuilder builder(&mem);

        const GraphResourceId lut = builder.createPersistentBuffer({
            .size = 256,
            .usage = BufferUsageFlags::Storage,
            .debugName = "Test.ReadOnlyPersistent" });

        const GraphNodeId a = builder.addComputePass("ReadA");
        builder.read(a, lut, ResourceUsage::StorageBuffer);
        const GraphNodeId b = builder.addComputePass("ReadB");
        builder.read(b, lut, ResourceUsage::StorageBuffer);

        FrameGraphCompiler compiler(&mem);
        const CompiledGraphPlan plan = compiler.compile(builder);

        check(findCrossFrame(plan, lut) == nullptr,
            "read-only persistent resource needs no cross-frame serialization");
    }

    void testHistoryPreviousReadEmitsCrossFrameEdge()
    {
        std::pmr::monotonic_buffer_resource mem;
        FrameGraphBuilder builder(&mem);

        const GraphResourceId history = builder.createHistoryTexture({
            .width = 64, .height = 64,
            .format = TextureFormat::RGBA8_UNorm,
            .usage = TextureUsageFlags::Sampled | TextureUsageFlags::Storage,
            .debugName = "Test.HistoryCrossFrame" });

        const GraphNodeId producer = builder.addComputePass("WriteCurrent");
        builder.write(producer, history, ResourceUsage::StorageTexture);
        const GraphNodeId consumer = builder.addComputePass("ReadPrevious");
        builder.readPrevious(consumer, history, ResourceUsage::SampledTexture);

        FrameGraphCompiler compiler(&mem);
        const CompiledGraphPlan plan = compiler.compile(builder);

        const CrossFrameDependency* dep = findCrossFrame(plan, history);
        check(dep != nullptr,
            "history previous-read emits a cross-frame edge");
        if (dep)
        {
            // The previous-instance reader waits for the prior frame's writer of
            // the instance it consumes.
            check(dep->consumerNode == consumer,
                "history cross-frame consumer is the previous-version reader");
        }
        // The current instance is per-frame-slot, so it must NOT additionally
        // force a WAR/WAW edge. Exactly one edge is generated for the resource.
        check(plan.crossFrameDependencies.size() == 1,
            "history resource emits only the previous-read edge");
    }
}

int runFrameGraphResourceTests()
{
    g_failures = 0;
    spdlog::warn("[FrameGraphResourceTests] running...");

    testInstanceIndexMath();
    testHistoryHasNoFalseDependency();
    testCurrentReadCreatesDependency();
    testMixedVersionGraphIsAcyclic();
    testGpuDrivenBuffersAreGraphOwned();
    testGpuCullReadsPreviousHiZ();
    testGpuCullDiagnosticsAreExplicitAndOptional();
    testOcclusionHistoryReliabilityFallbacks();
    testPerFrameSlotHasNoCrossFrameDependency();
    testPersistentWriteEmitsCrossFrameEdge();
    testReadOnlyPersistentHasNoCrossFrameEdge();
    testHistoryPreviousReadEmitsCrossFrameEdge();

    if (g_failures == 0)
    {
        spdlog::warn("[FrameGraphResourceTests] PASSED");
    }
    else
    {
        spdlog::error(
            "[FrameGraphResourceTests] {} check(s) FAILED", g_failures);
    }
    return g_failures;
}
