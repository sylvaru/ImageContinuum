#pragma once

//#include "ic/rendering/renderer_defs.h"
#include <span>
#include <vector>

#include "gpu_queue_profiler.h"
#include "ibl_baker.h"
#include "render_handles.h"
#include "ic/renderer/ray_tracing/ray_tracing_scene.h"

namespace ic 
{
    struct CompiledGraphPlan;
    struct GraphExecutionContext;
    struct FrameContext;
    struct SceneRenderView;
    struct GpuGiDiagnostics;

    struct RendererSpecification;

    class PipelineLibrary;
    class Window;

    // A Hi-Z pyramid mip, addressed as an ImGui-drawable handle. Kept
    // API-neutral so the diagnostics window can show the pyramid without
    // knowing which backend produced it.
    struct HiZDebugImage
    {
        uint64_t textureId = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t mipLevels = 0;
        bool valid = false;
    };

    // One backend feature and whether it is actually in use, as opposed to
    // merely advertised by the device -- the distinction that matters when
    // explaining why a path behaves differently on two machines.
    struct BackendFeature
    {
        const char* name = "";
        bool enabled = false;
        // Optional note, e.g. why an advertised feature is disabled.
        const char* detail = "";
    };

    struct BackendLimit
    {
        const char* name = "";
        uint64_t value = 0;
    };

    struct BackendDiagnosticInfo
    {
        const char* backendName = "";
        const char* adapterName = "";
        std::span<const BackendFeature> features;
        std::span<const BackendLimit> limits;
    };

    struct RendererPerformanceCounters
    {
        double backendCpuMs = 0.0;
        double frameSlotWaitMs = 0.0;
        double profilerReadbackMs = 0.0;
        double graphRecordMs = 0.0;
        double uiRecordMs = 0.0;
        double validationMs = 0.0;
        double submitPresentMs = 0.0;
    };

    // The authoritative render surface for a frame: the backend's swapchain
    // framebuffer extent plus a generation that increments on every swapchain
    // (re)creation. The renderer treats this, rather than the OS window size, as the
    // single source of truth for the render extent, so the compiled graph and
    // all materialized attachments always match the backbuffer. renderable is
    // false when the surface cannot be drawn to this frame (minimized, zero
    // area, or a failed reconcile); the renderer then skips the whole frame.
    struct RenderSurfaceState
    {
        uint32_t width = 0;
        uint32_t height = 0;
        uint64_t generation = 0;
        bool renderable = false;
    };

    class RendererBackend
    {
    public:
        virtual ~RendererBackend() = default;

        virtual void init(
            const RendererSpecification& spec,
            const PipelineLibrary& pipelineLibrary,
            Window& window,
            uint32_t workerCount
        ) = 0;
        virtual void shutdown() = 0;

        // Reconciles the swapchain to the current window framebuffer size,
        // recreating it (bumping the generation) when the framebuffer changed or
        // the swapchain is no longer valid. Called once per frame BEFORE the
        // renderer builds the graph, so the returned extent is stable for the
        // whole frame. Must not be called between record and present.
        [[nodiscard]] virtual RenderSurfaceState reconcileRenderSurface() = 0;

        [[nodiscard]] virtual bool execute(
            const CompiledGraphPlan& plan,
            const GraphExecutionContext& execution,
            const FrameContext& frame,
            const SceneRenderView& scene) = 0;

        //[[nodiscard]] virtual SwapchainInfo swapchainInfo() const = 0;

        virtual std::vector<IBLBakeResult> executeIBLBakeRequests(
            std::span<const IBLBakeRequest> requests,
            const FrameContext&)
        {
            std::vector<IBLBakeResult> results;
            results.reserve(requests.size());
            for (const IBLBakeRequest& request : requests)
            {
                results.push_back(
                    IBLBakeResult{
                        .requestId = request.requestId,
                        .handle = request.handle,
                        .success = false,
                        .error = "IBL baking is not implemented by this backend" });
            }

            return results;
        }

        virtual bool beginDebugGuiFrame()
        {
            return false;
        }

        virtual void endDebugGuiFrame()
        {
        }

        virtual bool vsyncEnabled() const
        {
            return true;
        }

        virtual void setVsyncEnabled(bool)
        {
        }

        virtual bool clusteredForwardHeatmapEnabled() const
        {
            return false;
        }

        virtual void setClusteredForwardHeatmapEnabled(bool)
        {
        }

        virtual bool hiZDebugViewEnabled() const
        {
            return false;
        }

        virtual void setHiZDebugViewEnabled(bool)
        {
        }

        virtual uint32_t hiZDebugMip() const
        {
            return 0;
        }

        virtual void setHiZDebugMip(uint32_t)
        {
        }

        virtual bool hiZDebugPrevious() const
        {
            return false;
        }

        virtual void setHiZDebugPrevious(bool)
        {
        }

        virtual bool gpuOcclusionEnabled() const
        {
            return true;
        }

        virtual void setGpuOcclusionEnabled(bool)
        {
        }

        virtual GpuCullDebugMode gpuCullDebugMode() const
        {
            return GpuCullDebugMode::Off;
        }

        virtual void setGpuCullDebugMode(GpuCullDebugMode)
        {
        }

        virtual GpuCullStats gpuCullStats() const
        {
            return {};
        }
        virtual GpuCullPerformance gpuCullPerformance() const
        {
            return {};
        }

        // Reports whether this backend can dispatch frame-graph compute passes on
        // a dedicated async queue that overlaps graphics work. When false, the
        // renderer keeps every compute pass on the graphics queue regardless of
        // the async toggle. Defaults to unsupported; real backends with a
        // separate compute queue override this.
        virtual bool supportsAsyncCompute() const
        {
            return false;
        }

        virtual RayTracingCapabilities rayTracingCapabilities() const
        {
            return {};
        }
        virtual bool globalIlluminationDiagnostics(
            GpuGiDiagnostics&) const { return false; }
        virtual void setGlobalIlluminationDisplay(
            uint32_t, float, float) {}
        virtual void setGlobalIlluminationRuntimeSettings(
            uint32_t, uint32_t, uint32_t, uint32_t) {}

        virtual void setRayTracingSceneService(RayTracingSceneService*)
        {
        }
        virtual void setRayTracingEnabled(bool) {}
        [[nodiscard]] virtual bool rayTracingEnabled() const { return false; }

        // Queue assignment is part of the compiled schedule. Before replacing
        // that schedule, the renderer calls this once so submissions made with
        // the old assignment can no longer reference its resources or sync
        // topology. This is deliberately not part of the per-frame path.
        virtual void drainForSchedulingTransition()
        {
        }

        // Per-pass GPU timestamps for the most recently completed frame, in a
        // single CPU-millisecond domain shared by every queue. Empty when the
        // backend has no timestamp support or profiling is off.
        //
        // This is the measured ground truth behind the async-compute decision:
        // whether a compute pass overlapped graphics can only be answered by
        // timestamps, never by the graph's structure (a pass scheduled on the
        // compute queue may still run entirely serialized against graphics).
        virtual std::span<const GpuPassSample> gpuPassSamples() const
        {
            return {};
        }

        virtual void setGpuProfilingEnabled(bool)
        {
        }

        [[nodiscard]] virtual bool gpuProfilingEnabled() const
        {
            return false;
        }

        // The Hi-Z pyramid mip as something ImGui can draw. Returning a handle
        // rather than drawing a window keeps the visualization inside the
        // consolidated diagnostics window instead of a backend-owned popup,
        // while the backend keeps ownership of the descriptor it hands out.
        // `textureId` is whatever the backend's ImGui binding expects
        // (D3D12 GPU descriptor handle / VkDescriptorSet).
        virtual HiZDebugImage hiZDebugImage(
            [[maybe_unused]] bool previous,
            [[maybe_unused]] uint32_t mip)
        {
            return {};
        }

        // Static description of the device: name, and the features/limits that
        // actually change how this renderer behaves. Backends build the storage
        // once at init and return spans into it, so reading this per frame is
        // free.
        virtual BackendDiagnosticInfo backendDiagnostics() const
        {
            return {};
        }

        virtual RendererPerformanceCounters performanceCounters() const
        {
            return {};
        }
    };
}
