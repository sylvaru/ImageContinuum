#pragma once

//#include "ic/rendering/renderer_defs.h"
#include "render_handles.h"

namespace ic 
{
    struct CompiledGraphPlan;
    struct FrameContext;
    struct RendererSpecification;

    class RendererBackend
    {
    public:
        virtual ~RendererBackend() = default;

        virtual void initialize(const RendererSpecification& spec) = 0;
        virtual void shutdown() = 0;

        virtual void execute(
            const CompiledGraphPlan& plan,
            const FrameContext& frame) = 0;

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

}