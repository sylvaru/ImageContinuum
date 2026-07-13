// ic/renderer/renderer.h
#pragma once
#include "ic/common/ic_common.h"
#include "ibl_baker.h"
#include "renderer_specification.h"
#include "ic/renderer/frame_graph/frame_graph_pass.h"
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
        [[nodiscard]] bool clusteredForwardHeatmapEnabled() const;
        void setClusteredForwardHeatmapEnabled(bool enabled);
        [[nodiscard]] bool hiZDebugViewEnabled() const;
        void setHiZDebugViewEnabled(bool enabled);
        [[nodiscard]] uint32_t hiZDebugMip() const;
        void setHiZDebugMip(uint32_t mip);
        [[nodiscard]] RenderPathType renderPathType() const;
        
        void buildOrRebuildFrameGraph();
        // Explicitly dirties passes whose execution policy subscribes to one
        // or more of these API-neutral reasons. Manual is suitable for tools
        // and one-off editor actions.
        void invalidatePasses(PassInvalidation reasons);

        [[nodiscard]] IBLHandle requestIBLBake(
            const IBLBakeDesc& desc);
        [[nodiscard]] IBLBakeState iblState(
            IBLHandle handle) const;
        [[nodiscard]] IBLProbeSnapshot iblSnapshot(
            IBLHandle handle) const;

        static Scope<RendererBackend> createBackend(
            RendererBackendType type);

        static Scope<RendererPath> createPath(RenderPathType type);

    private:

        void syncSceneEnvironmentIBL(
            FrameContext& frame,
            const SceneRenderView& scene);

        void processPendingRendererJobs(
            FrameContext& frame);
        void drawFrameGraphDebugWindow();

        struct Runtime;
        Scope<Runtime> m_runtime;

    };
}
