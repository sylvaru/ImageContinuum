#include "ic/common/ic_pch.h"
#include "ic/renderer/frame_graph/frame_graph_builder.h"
#include "ic/core/memory/frame_arena.h"
#include <algorithm>

namespace ic
{

	GraphResourceId
		FrameGraphBuilder::createTexture()
	{
		GraphResourceId id =
			static_cast<GraphResourceId>(
				m_resources.size());

		m_resources.push_back({
			.id = id,
			.type = GraphResourceType::Texture,
			.firstAccess = 0,
			.accessCount = 0
			});

		return id;
	}

	GraphResourceId
		FrameGraphBuilder::createBuffer()
	{
		GraphResourceId id =
			static_cast<GraphResourceId>(
				m_resources.size());

		m_resources.push_back({
			.id = id,
			.type = GraphResourceType::Buffer,
			.firstAccess = 0,
			.accessCount = 0
			});

		return id;
	}

	void FrameGraphBuilder::read(
			GraphNodeId node,
			GraphResourceId resource,
			ResourceUsage usage)
	{
		m_nodes[node].accesses.push_back({
			.node = node,
			.resource = resource,
			.access = AccessType::Read,
			.usage = usage
			});
	}

	void FrameGraphBuilder::write(
			GraphNodeId node,
			GraphResourceId resource,
			ResourceUsage usage)
	{
		m_nodes[node].accesses.push_back({
			.node = node,
			.resource = resource,
			.access = AccessType::Write,
			.usage = usage
			});
	}

	void FrameGraphBuilder::clear()
	{
		m_nodes.clear();
		m_resources.clear();
		m_payloads.clear();
	}
}