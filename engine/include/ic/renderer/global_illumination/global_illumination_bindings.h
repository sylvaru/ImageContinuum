#pragma once

#include <array>
#include <cstdint>

#include "ic/renderer/frame_graph/frame_graph_types.h"

namespace ic
{
    inline constexpr uint32_t GiBufferBindingCount = 13;
    inline constexpr uint32_t GiPrimaryInputBinding = 20;
    inline constexpr uint32_t GiHistoryInputBinding = 21;
    inline constexpr uint32_t GiSceneDepthBinding = 22;
    inline constexpr uint32_t GiSurfaceAttributesBinding = 23;
    inline constexpr uint32_t GiPreviousSceneDepthBinding = 24;
    inline constexpr uint32_t GiPreviousSurfaceAttributesBinding = 25;
    inline constexpr uint32_t GiOutputBinding = 30;
    inline constexpr uint32_t GiTlasBinding = 40;
    inline constexpr uint32_t GiGeometryTableBinding = 41;
    inline constexpr uint32_t GiInstanceTableBinding = 42;
    inline constexpr uint32_t GiMaterialTableBinding = 43;
    inline constexpr uint32_t GiMaxGeometryBufferCount = 256;
    inline constexpr uint32_t GiVertexBuffersBinding = 44;
    inline constexpr uint32_t GiIndexBuffersBinding =
        GiVertexBuffersBinding + GiMaxGeometryBufferCount;
    inline constexpr uint32_t GiBindlessTexturesBinding =
        GiIndexBuffersBinding + GiMaxGeometryBufferCount;
    inline constexpr uint32_t GiEnvironmentBinding =
        GiBindlessTexturesBinding + 4096;
    inline constexpr uint32_t GiBindlessSamplersBinding = 0;
    inline constexpr uint32_t GiEnvironmentSamplerBinding = 256;

    inline constexpr uint32_t GiVkVertexBuffersBinding = 44;
    inline constexpr uint32_t GiVkIndexBuffersBinding = 45;
    inline constexpr uint32_t GiVkBindlessTexturesBinding = 46;
    inline constexpr uint32_t GiVkBindlessSamplersBinding = 47;
    inline constexpr uint32_t GiVkEnvironmentBinding = 48;
    inline constexpr uint32_t GiVkEnvironmentSamplerBinding = 49;
    // Shared frame constants (camera + scene lights), CBV register b1 on DX12.
    inline constexpr uint32_t GiVkFrameConstantsBinding = 50;

    inline constexpr std::array<GraphResourceSemantic, GiBufferBindingCount>
        GiBufferSemantics = {
            GraphResourceSemantic::GiSurfaceData,
            GraphResourceSemantic::GiSurfels,
            GraphResourceSemantic::GiVisibilityMoments,
            GraphResourceSemantic::GiHashBuckets,
            GraphResourceSemantic::GiProbes,
            GraphResourceSemantic::GiAllocationQueue,
            GraphResourceSemantic::GiPriorityQueue,
            GraphResourceSemantic::GiIndirectArguments,
            GraphResourceSemantic::GiCounters,
            GraphResourceSemantic::GiDiagnostics,
            GraphResourceSemantic::GiResidualInterface,
            GraphResourceSemantic::GiHitRecords,
            GraphResourceSemantic::GiProbeStaging
        };
}
