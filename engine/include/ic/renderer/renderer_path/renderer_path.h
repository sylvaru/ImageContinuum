// ic/renderer/renderer_path/renderer_path.h
#pragma once

#include <cstdint>
#include "ic/renderer/frame_graph/frame_graph_types.h"


namespace ic
{
	class FrameGraphBuilder;
    class GlobalIlluminationSystem;

	struct RenderExtent
	{
		uint32_t width = 1;
		uint32_t height = 1;
	};

    enum class FrameGraphBuildReason : uint8_t
    {
        Startup,
        Resize,
        Explicit
    };

	struct RendererPathContext
	{
		RenderExtent renderExtent;
		FrameGraphBuildReason rebuildReason =
            FrameGraphBuildReason::Explicit;

        // When true, a path may place compute passes with no dependency on
        // graphics work onto QueueType::Compute so they overlap on a dedicated
        // GPU async queue. The renderer only sets this once the backend reports
        // supportsAsyncCompute() and the runtime toggle is enabled, so a path
        // can honor it unconditionally. When false, every compute pass stays on
        // the graphics queue and behavior matches the serial baseline.
        bool asyncComputeEnabled = false;
        bool occlusionDiagnosticsEnabled = false;
        GlobalIlluminationSystem* globalIllumination = nullptr;
        GraphResourceId rayTracingSceneToken = InvalidGraphResourceId;
	};

	class RendererPath
	{
	public:
		virtual ~RendererPath() = default;

		virtual void buildFrameGraph(
			RendererPathContext& ctx,
			FrameGraphBuilder& builder) = 0;
	};
}
