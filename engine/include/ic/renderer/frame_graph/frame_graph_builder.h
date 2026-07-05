#pragma once
#include <vector>
#include <span>
#include <string_view>
#include "frame_graph_types.h"
#include "frame_graph_pass.h"
#include "ic/renderer/render_pipeline.h"


namespace ic
{


	class FrameGraphBuilder
	{
	public:
        class GraphicsPassBuilder
        {
        public:
            GraphicsPassBuilder(
                FrameGraphBuilder& builder,
                GraphNodeId node)
                : m_builder(builder)
                , m_node(node)
            {}

            GraphicsPassBuilder& pipeline(std::string_view logicalName);
            GraphicsPassBuilder& drawList(DrawListKind kind);
            GraphicsPassBuilder& colorLoadOp(AttachmentLoadOp op);
            GraphicsPassBuilder& depthLoadOp(AttachmentLoadOp op);
            GraphicsPassBuilder& color(GraphResourceId resource);
            GraphicsPassBuilder& depth(GraphResourceId resource);

            operator GraphNodeId() const
            {
                return m_node;
            }

        private:
            FrameGraphBuilder& m_builder;
            GraphNodeId m_node;
        };

        class ComputePassBuilder
        {
        public:
            ComputePassBuilder(
                FrameGraphBuilder& builder,
                GraphNodeId node)
                : m_builder(builder)
                , m_node(node)
            {}

            ComputePassBuilder& pipeline(std::string_view logicalName);
            ComputePassBuilder& dispatch(
                uint32_t groupCountX,
                uint32_t groupCountY = 1,
                uint32_t groupCountZ = 1);
            ComputePassBuilder& read(
                GraphResourceId resource,
                ResourceUsage usage);
            ComputePassBuilder& write(
                GraphResourceId resource,
                ResourceUsage usage);

            operator GraphNodeId() const
            {
                return m_node;
            }

        private:
            FrameGraphBuilder& m_builder;
            GraphNodeId m_node;
        };

        class TransferPassBuilder
        {
        public:
            TransferPassBuilder(
                FrameGraphBuilder& builder,
                GraphNodeId node)
                : m_builder(builder)
                , m_node(node)
            {}

            TransferPassBuilder& read(
                GraphResourceId resource,
                ResourceUsage usage = ResourceUsage::TransferSrc);
            TransferPassBuilder& write(
                GraphResourceId resource,
                ResourceUsage usage = ResourceUsage::TransferDst);

            operator GraphNodeId() const
            {
                return m_node;
            }

        private:
            FrameGraphBuilder& m_builder;
            GraphNodeId m_node;
        };

		explicit FrameGraphBuilder(std::pmr::memory_resource* memory)
			: m_memory(memory)
			, m_nodes(memory)
			, m_resources(memory)
			, m_payloads(memory)
			, m_accesses(memory)
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
				}
				});

			return id;
		}

        GraphicsPassBuilder addGraphicsPass(std::string_view name);
        ComputePassBuilder addComputePass(std::string_view name);
        TransferPassBuilder addTransferPass(std::string_view name);

		// Resource creation
		GraphResourceId createTexture(const TextureDesc& desc = {});
		GraphResourceId createBuffer(const BufferDesc& desc = {});

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

        void setGraphicsPassPipeline(
            GraphNodeId node,
            PipelineId pipeline);

        void setGraphicsPassDrawList(
            GraphNodeId node,
            DrawListKind kind);

        void setGraphicsPassColorLoadOp(
            GraphNodeId node,
            AttachmentLoadOp op);

        void setGraphicsPassDepthLoadOp(
            GraphNodeId node,
            AttachmentLoadOp op);

        void setComputePassPipeline(
            GraphNodeId node,
            PipelineId pipeline);

        void setComputePassDispatch(
            GraphNodeId node,
            uint32_t groupCountX,
            uint32_t groupCountY,
            uint32_t groupCountZ);

	private:

		std::pmr::vector<GraphResource> m_resources;
		std::pmr::vector<NodeRecord> m_nodes;
		std::pmr::vector<PassPayload> m_payloads;
		std::pmr::vector<ResourceAccess> m_accesses;

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

		[[nodiscard]]
		std::span<const ResourceAccess>
			accesses() const noexcept
		{
			return m_accesses;
		}
	};
}
