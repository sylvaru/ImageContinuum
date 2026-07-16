#include "common/tests_pch.h"
#include "pass_dispatch_tests.h"

#include "ic/renderer/frame_graph/frame_graph_pass.h"

#include <spdlog/spdlog.h>

#include <variant>

using namespace ic;

namespace
{
    int g_failures = 0;

    void check(bool condition, const char* what)
    {
        if (!condition)
        {
            ++g_failures;
            spdlog::error("[PassDispatchTests] FAIL: {}", what);
        }
    }

    // The recorder a payload routes to. This mirrors the categories both
    // backends' recordPassPayload overload sets resolve to: real recorders for
    // the passes that record work, and NoOp for the declared-but-unused feature
    // payloads plus the graph-only Clear/Present passes.
    enum class RecorderKind
    {
        Graphics,
        Compute,
        PathTrace,
        Tonemap,
        EnvironmentConvert,
        Transfer,
        NoOp
    };

    // Exhaustive payload -> recorder-kind visitor. Deliberately an explicit
    // overload per PassPayload alternative with NO generic/catch-all case, so it
    // is the same compile-time exhaustiveness guarantee the backends rely on:
    // adding a PassPayload alternative without a routing decision fails to
    // compile here (and in dispatchNode) rather than silently no-oping. Keep this
    // in lockstep with DX12Backend/VulkanBackend::recordPassPayload.
    struct RouteVisitor
    {
        RecorderKind operator()(const GraphicsPassData&) const
        {
            return RecorderKind::Graphics;
        }
        RecorderKind operator()(const ComputePassData&) const
        {
            return RecorderKind::Compute;
        }
        RecorderKind operator()(const PathTracePassData&) const
        {
            return RecorderKind::PathTrace;
        }
        RecorderKind operator()(const TonemapPassData&) const
        {
            return RecorderKind::Tonemap;
        }
        RecorderKind operator()(const EnvironmentConvertPassData&) const
        {
            return RecorderKind::EnvironmentConvert;
        }
        RecorderKind operator()(const TransferPassData&) const
        {
            return RecorderKind::Transfer;
        }
        RecorderKind operator()(const GeometryPassData&) const
        {
            return RecorderKind::NoOp;
        }
        RecorderKind operator()(const LightingPassData&) const
        {
            return RecorderKind::NoOp;
        }
        RecorderKind operator()(const ShadowPassData&) const
        {
            return RecorderKind::NoOp;
        }
        RecorderKind operator()(const PostProcessPassData&) const
        {
            return RecorderKind::NoOp;
        }
        RecorderKind operator()(const ClearPassData&) const
        {
            return RecorderKind::NoOp;
        }
        RecorderKind operator()(const PresentPassData&) const
        {
            return RecorderKind::NoOp;
        }
    };

    RecorderKind route(const PassPayload& payload)
    {
        return std::visit(RouteVisitor{}, payload);
    }

    // ---- Property: every payload routes to the intended recorder ----------
    void testPayloadRouting()
    {
        check(route(PassPayload{GraphicsPassData{}}) == RecorderKind::Graphics,
            "GraphicsPassData -> Graphics recorder");
        check(route(PassPayload{ComputePassData{}}) == RecorderKind::Compute,
            "ComputePassData -> Compute recorder");
        check(route(PassPayload{PathTracePassData{}}) == RecorderKind::PathTrace,
            "PathTracePassData -> PathTrace recorder");
        check(route(PassPayload{TonemapPassData{}}) == RecorderKind::Tonemap,
            "TonemapPassData -> Tonemap recorder");
        check(route(PassPayload{EnvironmentConvertPassData{}}) ==
                  RecorderKind::EnvironmentConvert,
            "EnvironmentConvertPassData -> EnvironmentConvert recorder");
        check(route(PassPayload{TransferPassData{}}) == RecorderKind::Transfer,
            "TransferPassData -> Transfer recorder");
        check(route(PassPayload{PresentPassData{}}) == RecorderKind::NoOp,
            "PresentPassData is graph-only (NoOp)");
        check(route(PassPayload{ClearPassData{}}) == RecorderKind::NoOp,
            "ClearPassData is graph-only (NoOp)");
        check(route(PassPayload{GeometryPassData{}}) == RecorderKind::NoOp,
            "unused GeometryPassData routes to NoOp");
        check(route(PassPayload{LightingPassData{}}) == RecorderKind::NoOp,
            "unused LightingPassData routes to NoOp");
        check(route(PassPayload{ShadowPassData{}}) == RecorderKind::NoOp,
            "unused ShadowPassData routes to NoOp");
        check(route(PassPayload{PostProcessPassData{}}) == RecorderKind::NoOp,
            "unused PostProcessPassData routes to NoOp");
    }

    // ---- Property: the visitor covers exactly the variant's alternatives ---
    //
    // std::visit already fails to compile if RouteVisitor is not callable for
    // every alternative. This guards the other direction the compiler can't:
    // that the alternative COUNT still matches what the routing test enumerates,
    // so adding a payload forces both this count and a routing case to be
    // revisited rather than defaulting an unclassified pass to some neighbour.
    void testVariantAlternativeCount()
    {
        check(std::variant_size_v<PassPayload> == 12,
            "PassPayload alternative count matches the dispatch routing table");
    }
}

int runPassDispatchTests()
{
    g_failures = 0;
    spdlog::warn("[PassDispatchTests] running...");

    testPayloadRouting();
    testVariantAlternativeCount();

    if (g_failures == 0)
    {
        spdlog::warn("[PassDispatchTests] PASSED");
    }
    else
    {
        spdlog::error(
            "[PassDispatchTests] {} check(s) FAILED", g_failures);
    }
    return g_failures;
}
