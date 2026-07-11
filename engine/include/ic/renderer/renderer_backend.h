#pragma once

//#include "ic/rendering/renderer_defs.h"
#include <span>
#include <vector>

#include "ibl_baker.h"
#include "render_handles.h"

namespace ic 
{
    struct CompiledGraphPlan;
    struct FrameContext;
    struct SceneRenderView;

    struct RendererSpecification;

    class PipelineLibrary;
    class Window;

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

        virtual void execute(
            const CompiledGraphPlan& plan,
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
    };
}
