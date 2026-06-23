#pragma once

//#include "ic/rendering/renderer_defs.h"
#include "render_handles.h"

namespace ic 
{

    class RendererBackend
    {
    public:
        virtual ~RendererBackend() = default;

        virtual void initialize(const struct RendererSpecification& spec) = 0;
        virtual void shutdown() = 0;

        // Opaque backend management methods acting entirely on handles
        virtual void immediate_create_bffer(
            BufferHandle handle, uint64_t size) = 0;

        virtual void submit_frame_commands(
            const class CompiledFramePlan& plan) = 0;

    };

}