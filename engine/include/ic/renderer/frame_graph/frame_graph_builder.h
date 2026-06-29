#pragma once
#include <vector>
#include <span>
#include "frame_graph_types.h"
#include "frame_graph_pass.h"


namespace ic
{


	class FrameGraphBuilder
	{
	public:
		explicit FrameGraphBuilder(std::pmr::memory_resource* memory)
			: m_memory(memory)
			, m_nodes(memory)
			, m_resources(memory)
			, m_payloads(memory)
		{}

		~FrameGraphBuilder() = default;

		// Node creation
		template<typename T>
		GraphNodeId addGraphNode(
			const T& payload,
			GraphNodeType type,
			QueueType queue)
		{
			const uint32_t payloadIndex =
				static_cast<uint32_t>(
					m_payloads.size());

			m_payloads.emplace_back(payload);

			GraphNodeId id =
				static_cast<GraphNodeId>(
					m_nodes.size());

			m_nodes.push_back(NodeRecord{
				.graphNode = {
					.id = id,
					.queue = queue,
					.type = type,
					.payloadIndex = payloadIndex
				},
				.accesses = {}
				});

			return id;
		}

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

		GraphResourceId importTexture(ImportedResourceDesc desc);

		void clear();

	private:

		std::pmr::vector<GraphResource> m_resources;
		std::pmr::vector<NodeRecord> m_nodes;
		std::pmr::vector<PassPayload> m_payloads;

		std::pmr::memory_resource* m_memory;
	public:

		std::pmr::memory_resource* 
			get_allocator() const noexcept
		{
			return m_memory;
		}


		[[nodiscard]]
		std::span<const GraphResource>
			resources() const noexcept
		{
			return m_resources;
		}

		[[nodiscard]]
		std::span<const NodeRecord>
			nodes() const noexcept
		{
			return m_nodes;
		}


		[[nodiscard]]
		std::span<const PassPayload>
			payloads() const noexcept
		{
			return m_payloads;
		}
	};
}