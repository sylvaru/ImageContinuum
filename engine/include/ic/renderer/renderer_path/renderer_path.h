// ic/renderer/renderer_path/renderer_path.h
#pragma once


namespace ic
{
	class FrameGraphBuilder;

	class RendererPath
	{
	public:
		virtual ~RendererPath() = default;

		virtual void buildFrameGraph(
			FrameGraphBuilder& builder) = 0;
		
		virtual void buildPassData() {}
	};
}