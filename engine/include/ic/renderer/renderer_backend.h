#pragma once

//#include "ic/rendering/renderer_defs.h"
#include <span>
#include <vector>

#include "ibl_baker.h"
#include "render_handles.h"

namespace ic 
{
    struct CompiledGraphPlan;
    struct GraphExecutionContext;
    struct FrameContext;
    struct SceneRenderView;

    struct RendererSpecification;

    class PipelineLibrary;
    class Window;

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

        // Reports whether this backend can dispatch frame-graph compute passes on
        // a dedicated async queue that overlaps graphics work. When false, the
        // renderer keeps every compute pass on the graphics queue regardless of
        // the async toggle. Defaults to unsupported; real backends with a
        // separate compute queue override this.
        virtual bool supportsAsyncCompute() const
        {
            return false;
        }
    };
}
