// ic/renderer/renderer_path/renderer_path.h
#pragma once


namespace ic
{
	class FrameGraphBuilder;

	struct RenderExtent
	{
		uint32_t width = 1;
		uint32_t height = 1;
	};

	struct RendererPathContext
	{
		RenderExtent renderExtent;
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