// ic/renderer/renderer_path/renderer_path.h
#pragma once


namespace ic
{
	class FrameGraphBuilder;
	struct RenderScenePacket;
	
	// Todo: All 3 of these render paths should 
	// be using visibility buffers eventually
	enum class RenderPath : uint8_t
	{
		ClusteredForward,
		Deferred,
		PathTraced
	};

	class RendererPath
	{
	public:
		virtual ~RendererPath() = default;

		virtual void setupGraph(
			FrameGraphBuilder& builder,
			const RenderScenePacket& packet) = 0;
	};
}