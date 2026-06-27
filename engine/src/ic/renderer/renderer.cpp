// ic/renderer/renderer.h
#include "ic/common/ic_pch.h"
#include "ic/renderer/renderer.h"
#include "ic/renderer/renderer_backend.h"
#include "ic/renderer/vulkan_backend/vulkan_backend.h"
#include "ic/renderer/renderer_path/renderer_path.h"
#include "ic/renderer/renderer_path/forward_path.h"
#include "ic/renderer/frame_graph/frame_graph_builder.h"
#include "ic/renderer/frame_graph/frame_graph_compiler.h"
#include "ic/renderer/frame_graph/frame_graph_arena.h"
#include "ic/renderer/frame_graph/compiled_graph_plan.h"


namespace ic
{

    struct Renderer::Runtime
    {
        FrameGraphArena arena;

        CompiledGraphPlan compiledGraphPlan;

        Scope<RendererBackend> backend;
        Scope<RendererPath> path;

        FrameGraphBuilder builder;
        FrameGraphCompiler compiler;

        bool graphDirty = true;

        Runtime(
            Scope<RendererBackend> b,
            Scope<RendererPath> p)
            : backend(std::move(b))
            , path(std::move(p))
            , builder(arena.resource())
            , compiler(arena.resource())
        {
        }
    };

    Renderer::Renderer(const RendererSpecification& spec)
    {
        m_runtime = std::make_unique<Runtime>(
            createBackend(spec.backendType),
            createPath(spec.pathType)
        );
    }
    Renderer::~Renderer()
    {}

    void Renderer::init(RendererSpecification& spec)
    {
        spdlog::info("[Renderer] init...");

        m_runtime->backend->initialize(spec);

        rebuildGraph();
    }

    void Renderer::render(
        [[maybe_unused]] FrameContext& frame)
    {
        auto& rt = *m_runtime;

        if (rt.graphDirty)
        {
            rebuildGraph();
        }

        rt.backend->execute(
            rt.compiledGraphPlan,
            frame);
    }

    void Renderer::rebuildGraph()
    {
        auto& rt = *m_runtime;
        //rt.arena.reset();

        rt.builder.clear();

        rt.path->buildFrameGraph(rt.builder);

        rt.compiledGraphPlan = rt.compiler.compile(rt.builder);

        rt.graphDirty = false;
    }

    Scope<RendererBackend> 
        Renderer::createBackend(RendererBackendType type)
    {
        switch (type)
        {
        case RendererBackendType::Vulkan:
            return std::make_unique<VulkanBackend>();

        //case RendererBackendType::DX12:
        //    return std::make_unique<DX12Backend>();
        }

        return nullptr;
    }

    Scope<RendererPath> 
        Renderer::createPath(RenderPathType type)
    {
        switch (type)
        {
        case RenderPathType::Forward:
            return std::make_unique<ForwardRendererPath>();

            //case RenderPathType::Deferred:
            //    return std::make_unique<DeferredRendererPath>();

            //case RenderPathType::PathTraced:
            //    return std::make_unique<PathTracerRendererPath>();
        }

        return nullptr;
    }




}