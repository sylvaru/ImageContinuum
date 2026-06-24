// ic/renderer/frame_graph/frame_graph.h
#pragma once
#include <span>

namespace ic
{

	struct FrameNode;
	struct ResourceAccess;
	struct FrameResource;
	/*
		This Frame graph is API agnostic. 
		It's pure data oriented logic. 
		It treats resources as handles
	
	*/

	struct FrameGraph
	{
		std::span<FrameNode> frameNode;
		std::span<ResourceAccess> resourceAccess;
		std::span<FrameResource> frameResource;
	};
}