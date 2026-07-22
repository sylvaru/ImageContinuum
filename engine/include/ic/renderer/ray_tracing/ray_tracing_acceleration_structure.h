#pragma once

#include <cstdint>

#include "ic/renderer/ray_tracing/ray_tracing_scene.h"

namespace ic
{
    enum class RayTracingAccelerationStructureState : uint8_t
    {
        Disabled,
        Unsupported,
        Empty,
        BuildPending,
        Ready,
        Failed
    };

    struct RayTracingAccelerationStructureStatistics
    {
        uint64_t sceneGeneration = 0;
        uint64_t blasBuildCount = 0;
        uint64_t blasReuseCount = 0;
        uint64_t blasRefitCount = 0;
        uint64_t tlasBuildCount = 0;
        uint64_t tlasRefitCount = 0;
        uint64_t retiredResourceCount = 0;
        uint32_t blasCount = 0;
        uint32_t instanceCount = 0;
        uint64_t resultBytes = 0;
        uint64_t scratchBytes = 0;
        RayTracingSceneUpdateKind lastUpdate =
            RayTracingSceneUpdateKind::None;
        RayTracingAccelerationStructureState state =
            RayTracingAccelerationStructureState::Unsupported;
    };

    class RayTracingAccelerationStructureProvider
    {
    public:
        virtual ~RayTracingAccelerationStructureProvider() = default;
        virtual void setEnabled(bool enabled) = 0;
        [[nodiscard]] virtual bool enabled() const noexcept = 0;
        virtual void invalidate() noexcept = 0;
        [[nodiscard]] virtual RayTracingCapabilities capabilities()
            const noexcept = 0;
        [[nodiscard]] virtual bool readyFor(
            uint64_t sceneGeneration) const noexcept = 0;
        [[nodiscard]] virtual uint64_t shaderTlasHandle() const noexcept = 0;
        [[nodiscard]] virtual const
            RayTracingAccelerationStructureStatistics& statistics()
            const noexcept = 0;
    };
}
