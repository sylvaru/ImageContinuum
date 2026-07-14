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
            GraphicsPassBuilder& cadence(PassCadence cadence);
            GraphicsPassBuilder& onInvalidation(PassInvalidation reasons);
            GraphicsPassBuilder& once();

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
            ComputePassBuilder& cadence(PassCadence cadence);
            ComputePassBuilder& onInvalidation(PassInvalidation reasons);
            ComputePassBuilder& once();

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
            TransferPassBuilder& copy(
                GraphResourceId source,
                GraphResourceId destination);
            TransferPassBuilder& queue(QueueType queue);
            TransferPassBuilder& cadence(PassCadence cadence);
            TransferPassBuilder& onInvalidation(PassInvalidation reasons);
            TransferPassBuilder& once();

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

        // A compute pass defaults to the graphics queue.
        //
        // Every compute pass the renderer currently records (GPU cull, Hi-Z
        // build, cluster build, light culling) is a link in a serial chain that
        // both consumes and feeds graphics work in the same frame
        // (depth -> Hi-Z -> cull -> cluster -> opaque shading), so an async
        // compute queue has nothing to overlap and buys no parallelism.
        //
        // It would also be unsafe as written: several of these passes touch
        // backend-managed resources (scene depth, the cluster/light buffers)
        // that live outside the frame graph's barrier tracking, and D3D12
        // requires any resource handed between the graphics and compute queues
        // to pass through the COMMON state at the boundary. That cross-queue
        // COMMON handoff is not implemented for those non-graph resources, and
        // omitting it is what triggers DXGI_ERROR_ACCESS_DENIED device removal.
        //
        // Passing QueueType::Compute explicitly opts a pass into the async queue
        // and is only correct once a workload has independent work to overlap
        // AND the cross-queue COMMON handoff above is implemented for every
        // resource it shares with another queue.
        ComputePassBuilder addComputePass(
            std::string_view name,
            QueueType queue = QueueType::Graphics);

        TransferPassBuilder addTransferPass(std::string_view name);

		// Resource creation
		GraphResourceId createTexture(const TextureDesc& desc = {});
		GraphResourceId createBuffer(const BufferDesc& desc = {});
        GraphResourceId createPersistentTexture(const TextureDesc& desc = {});
        GraphResourceId createPersistentBuffer(const BufferDesc& desc = {});
        void setResourceSemantic(
            GraphResourceId resource,
            GraphResourceSemantic semantic);

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

        void setTransferCopy(
            GraphNodeId node,
            GraphResourceId source,
            GraphResourceId destination);
        void setNodeQueue(GraphNodeId node, QueueType queue);

        void setPassCadence(
            GraphNodeId node,
            PassCadence cadence);
        void setPassInvalidation(
            GraphNodeId node,
            PassInvalidation reasons);

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
