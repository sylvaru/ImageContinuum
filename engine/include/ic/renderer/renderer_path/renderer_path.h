// ic/renderer/renderer_path/renderer_path.h
#pragma once

#include <cstdint>


namespace ic
{
	class FrameGraphBuilder;

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
