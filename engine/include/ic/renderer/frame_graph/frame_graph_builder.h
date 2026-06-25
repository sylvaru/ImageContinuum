#pragma once
#include <vector>
#include <span>
#include "frame_graph_types.h"


namespace ic
{


	class FrameGraphBuilder
	{
	public:
		explicit FrameGraphBuilder(std::pmr::memory_resource* memory)
			: m_nodes(memory)
			, m_resources(memory)
			, m_reads(memory)
			, m_writes(memory)
		{}

		~FrameGraphBuilder() = default;

		// Node creation
		GraphNodeId addGraphNode(
			GraphNodeType type,
			QueueType queue = QueueType::Graphics);


		// Resource creation
		GraphResourceId createTexture(); // TextureDesc desc
		GraphResourceId createBuffer(); // BufferDesc desc

		// Dependency declaration
		void read(
			GraphNodeId node,
			GraphResourceId resource,
			ResourceUsage usage);

		void write(
			GraphNodeId node,
			GraphResourceId resource,
			ResourceUsage usage);

		void clear();

		[[nodiscard]]
		std::span<const GraphNode>
			nodes() const noexcept
		{
			return m_nodes;
		}

		[[nodiscard]]
		std::span<const GraphResource>
			resources() const noexcept
		{
			return m_resources;
		}

		[[nodiscard]]
		std::span<const ResourceAccess>
			reads() const noexcept
		{
			return m_reads;
		}

		[[nodiscard]]
		std::span<const ResourceAccess>
			writes() const noexcept
		{
			return m_writes;
		}

	private:

		std::pmr::vector<GraphNode> m_nodes;
		std::pmr::vector<GraphResource> m_resources;
		std::pmr::vector<ResourceAccess> m_reads;
		std::pmr::vector<ResourceAccess> m_writes;
	};
}