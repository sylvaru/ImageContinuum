#pragma once

//#include "ic/rendering/renderer_defs.h"
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

        virtual void initialize(
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

        // Opaque backend management methods acting entirely on handles
        //virtual void createBuffer(
        //    BufferHandle,
        //    const BufferDesc&) = 0;

        //virtual void createTexture(
        //    TextureHandle,
        //    const TextureDesc&) = 0;

        //virtual void createPipeline(
        //    PipelineHandle,
        //    const PipelineDesc&) = 0;
    };

    // Backend should eventually know almost nothing about how dependencies were generated
    //for (GraphNodeId nodeId : plan.executionOrder)
    //{
    //    executeNode(nodeId);
    //}
}
