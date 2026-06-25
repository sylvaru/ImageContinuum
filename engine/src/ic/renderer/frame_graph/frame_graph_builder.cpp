#include "ic/common/ic_pch.h"
#include "ic/renderer/frame_graph/frame_graph_builder.h"
#include "ic/core/memory/frame_arena.h"
#include <algorithm>

namespace ic
{
	GraphNodeId
		FrameGraphBuilder::addGraphNode(
			GraphNodeType type,
			QueueType queue)
	{
		GraphNodeId id =
			static_cast<GraphNodeId>(
				m_nodes.size());

		m_nodes.push_back({
			.id = id,
			.queue = queue,
			.type = type,
			.firstRead = 0,
			.readCount = 0,
			.firstWrite = 0,
			.writeCount = 0,
			.payloadIndex = 0
			});

		return id;
	}

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
		m_reads.push_back({
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
		m_writes.push_back({
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
		m_reads.clear();
		m_writes.clear();
	}
}