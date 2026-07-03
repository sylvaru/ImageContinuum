// ic/renderer/renderer.h
#pragma once
#include "ic/common/ic_common.h"
#include "renderer_specification.h"
namespace ic
{
    struct FrameContext;
    struct RendererSpecification;
    struct SceneRenderView;

    class Window;
    class RendererBackend;
    class RendererPath;


    class Renderer
    {
    public:
        Renderer(
            const RendererSpecification& spec);

        ~Renderer();


        void init(
            RendererSpecification& spec,
            Window& window,
            uint32_t workerCount
        );
        void shutdown();

        void render(FrameContext& frame);
        void render(
            FrameContext& frame,
            const SceneRenderView& scene);
        bool beginDebugGuiFrame();
        void endDebugGuiFrame();
        [[nodiscard]] bool vsyncEnabled() const;
        void setVsyncEnabled(bool enabled);
        
        void rebuildGraph();

        static Scope<RendererBackend> createBackend(
            RendererBackendType type);

        static Scope<RendererPath> createPath(RenderPathType type);

    private:
        struct Runtime;
        Scope<Runtime> m_runtime;

    };
}
