#include "ic/common/ic_pch.h"
#include "ic/renderer/frame_graph/frame_graph_builder.h"
#include "ic/core/frame_memory/frame_arena.h"
#include <algorithm>

namespace ic
{
    FrameGraphBuilder::GraphicsPassBuilder&
        FrameGraphBuilder::GraphicsPassBuilder::pipeline(
            std::string_view logicalName)
    {
        m_builder.setGraphicsPassPipeline(
            m_node,
            makePipelineId(logicalName));
        return *this;
    }

    FrameGraphBuilder::GraphicsPassBuilder&
        FrameGraphBuilder::GraphicsPassBuilder::drawList(
            DrawListKind kind)
    {
        m_builder.setGraphicsPassDrawList(
            m_node,
            kind);
        return *this;
    }

    FrameGraphBuilder::GraphicsPassBuilder&
        FrameGraphBuilder::GraphicsPassBuilder::colorLoadOp(
            AttachmentLoadOp op)
    {
        m_builder.setGraphicsPassColorLoadOp(m_node, op);
        return *this;
    }

    FrameGraphBuilder::GraphicsPassBuilder&
        FrameGraphBuilder::GraphicsPassBuilder::depthLoadOp(
            AttachmentLoadOp op)
    {
        m_builder.setGraphicsPassDepthLoadOp(m_node, op);
        return *this;
    }

    FrameGraphBuilder::GraphicsPassBuilder&
        FrameGraphBuilder::GraphicsPassBuilder::color(
            GraphResourceId resource)
    {
        m_builder.write(
            m_node,
            resource,
            ResourceUsage::ColorAttachment);
        return *this;
    }

    FrameGraphBuilder::GraphicsPassBuilder&
        FrameGraphBuilder::GraphicsPassBuilder::cadence(
            PassCadence cadence)
    {
        m_builder.setPassCadence(m_node, cadence);
        return *this;
    }

    FrameGraphBuilder::GraphicsPassBuilder&
        FrameGraphBuilder::GraphicsPassBuilder::onInvalidation(
            PassInvalidation reasons)
    {
        m_builder.setPassInvalidation(m_node, reasons);
        return *this;
    }

    FrameGraphBuilder::GraphicsPassBuilder&
        FrameGraphBuilder::GraphicsPassBuilder::once()
    {
        return cadence(PassCadence::Once);
    }

    FrameGraphBuilder::ComputePassBuilder&
        FrameGraphBuilder::ComputePassBuilder::cadence(
            PassCadence cadence)
    {
        m_builder.setPassCadence(m_node, cadence);
        return *this;
    }

    FrameGraphBuilder::ComputePassBuilder&
        FrameGraphBuilder::ComputePassBuilder::onInvalidation(
            PassInvalidation reasons)
    {
        m_builder.setPassInvalidation(m_node, reasons);
        return *this;
    }

    FrameGraphBuilder::ComputePassBuilder&
        FrameGraphBuilder::ComputePassBuilder::once()
    {
        return cadence(PassCadence::Once);
    }

    FrameGraphBuilder::TransferPassBuilder&
        FrameGraphBuilder::TransferPassBuilder::cadence(
            PassCadence cadence)
    {
        m_builder.setPassCadence(m_node, cadence);
        return *this;
    }

    FrameGraphBuilder::TransferPassBuilder&
        FrameGraphBuilder::TransferPassBuilder::onInvalidation(
            PassInvalidation reasons)
    {
        m_builder.setPassInvalidation(m_node, reasons);
        return *this;
    }

    FrameGraphBuilder::TransferPassBuilder&
        FrameGraphBuilder::TransferPassBuilder::once()
    {
        return cadence(PassCadence::Once);
    }

    FrameGraphBuilder::GraphicsPassBuilder&
        FrameGraphBuilder::GraphicsPassBuilder::depth(
            GraphResourceId resource)
    {
        m_builder.write(
            m_node,
            resource,
            ResourceUsage::DepthAttachment);
        return *this;
    }

    FrameGraphBuilder::GraphicsPassBuilder
        FrameGraphBuilder::addGraphicsPass(std::string_view name)
    {
        GraphicsPassData data{};
        data.name = name;

        const GraphNodeId node =
            addGraphNode(
                data,
                GraphNodeType::Graphics,
                QueueType::Graphics);

        return GraphicsPassBuilder(*this, node);
    }

    FrameGraphBuilder::ComputePassBuilder&
        FrameGraphBuilder::ComputePassBuilder::pipeline(
            std::string_view logicalName)
    {
        m_builder.setComputePassPipeline(
            m_node,
            makePipelineId(logicalName));
        return *this;
    }

    FrameGraphBuilder::ComputePassBuilder&
        FrameGraphBuilder::ComputePassBuilder::dispatch(
            uint32_t groupCountX,
            uint32_t groupCountY,
            uint32_t groupCountZ)
    {
        m_builder.setComputePassDispatch(
            m_node,
            groupCountX,
            groupCountY,
            groupCountZ);
        return *this;
    }

    FrameGraphBuilder::ComputePassBuilder&
        FrameGraphBuilder::ComputePassBuilder::read(
            GraphResourceId resource,
            ResourceUsage usage)
    {
        m_builder.read(m_node, resource, usage);
        return *this;
    }

    FrameGraphBuilder::ComputePassBuilder&
        FrameGraphBuilder::ComputePassBuilder::write(
            GraphResourceId resource,
            ResourceUsage usage)
    {
        m_builder.write(m_node, resource, usage);
        return *this;
    }

    FrameGraphBuilder::ComputePassBuilder
        FrameGraphBuilder::addComputePass(std::string_view name, QueueType queue)
    {
        ComputePassData data{};
        data.name = name;

        const GraphNodeId node =
            addGraphNode(
                data,
                GraphNodeType::Compute,
                queue);

        return ComputePassBuilder(*this, node);
    }

    FrameGraphBuilder::TransferPassBuilder&
        FrameGraphBuilder::TransferPassBuilder::read(
            GraphResourceId resource,
            ResourceUsage usage)
    {
        m_builder.read(m_node, resource, usage);
        return *this;
    }

    FrameGraphBuilder::TransferPassBuilder&
        FrameGraphBuilder::TransferPassBuilder::write(
            GraphResourceId resource,
            ResourceUsage usage)
    {
        m_builder.write(m_node, resource, usage);
        return *this;
    }

    FrameGraphBuilder::TransferPassBuilder&
        FrameGraphBuilder::TransferPassBuilder::copy(
            GraphResourceId source,
            GraphResourceId destination)
    {
        m_builder.setTransferCopy(m_node, source, destination);
        m_builder.read(m_node, source, ResourceUsage::TransferSrc);
        m_builder.write(m_node, destination, ResourceUsage::TransferDst);
        return *this;
    }

    FrameGraphBuilder::TransferPassBuilder&
        FrameGraphBuilder::TransferPassBuilder::queue(QueueType queue)
    {
        m_builder.setNodeQueue(m_node, queue);
        return *this;
    }

    FrameGraphBuilder::TransferPassBuilder
        FrameGraphBuilder::addTransferPass(std::string_view name)
    {
        TransferPassData data{};
        data.name = name;

        const GraphNodeId node =
            addGraphNode(
                data,
                GraphNodeType::Transfer,
                QueueType::Transfer);

        return TransferPassBuilder(*this, node);
    }

	GraphResourceId
		FrameGraphBuilder::createTexture(const TextureDesc& desc)
	{
		GraphResourceId id =
			static_cast<GraphResourceId>(
				m_resources.size());

		m_resources.push_back({
			.id = id,
			.type = GraphResourceType::Texture,
			.ownership = ResourceOwnership::Transient,
            .multiplicity = ResourceMultiplicity::PerFrameSlot,
            .initialUsage = ResourceUsage::StorageTexture,
            .initialAccess = AccessType::Write,
            .textureDesc = desc,
			.firstAccess = 0,
			.accessCount = 0
			});

		return id;
	}

	GraphResourceId
		FrameGraphBuilder::createBuffer(const BufferDesc& desc)
	{
		GraphResourceId id =
			static_cast<GraphResourceId>(
				m_resources.size());

		m_resources.push_back({
			.id = id,
			.type = GraphResourceType::Buffer,
			.ownership = ResourceOwnership::Transient,
            .multiplicity = ResourceMultiplicity::PerFrameSlot,
            .initialUsage = ResourceUsage::StorageBuffer,
            .initialAccess = AccessType::Write,
            .bufferDesc = desc,
			.firstAccess = 0,
			.accessCount = 0
			});

		return id;
	}

    GraphResourceId FrameGraphBuilder::createPersistentTexture(
        const TextureDesc& desc)
    {
        const GraphResourceId id = createTexture(desc);
        m_resources[id].ownership = ResourceOwnership::Persistent;
        // Persistent resources are a single physical instance whose cross-frame
        // reuse is ordered by the executor, not double-buffered.
        m_resources[id].multiplicity = ResourceMultiplicity::Single;
        return id;
    }

    GraphResourceId FrameGraphBuilder::createPersistentBuffer(
        const BufferDesc& desc)
    {
        const GraphResourceId id = createBuffer(desc);
        m_resources[id].ownership = ResourceOwnership::Persistent;
        m_resources[id].multiplicity = ResourceMultiplicity::Single;
        return id;
    }

    GraphResourceId FrameGraphBuilder::createHistoryTexture(
        const TextureDesc& desc)
    {
        // A history resource is transient storage (recreated with the graph)
        // but backed by one instance per frame-in-flight slot, so the previous
        // frame's contents remain intact while the current frame writes a new
        // instance. Consumers reach last frame's instance via the registry's
        // previousEntry(); the executor's cross-frame ordering guarantees the
        // previous producer has completed before this frame reads it.
        const GraphResourceId id = createTexture(desc);
        m_resources[id].multiplicity = ResourceMultiplicity::History;
        return id;
    }

    GraphResourceId FrameGraphBuilder::createHistoryBuffer(
        const BufferDesc& desc)
    {
        const GraphResourceId id = createBuffer(desc);
        m_resources[id].multiplicity = ResourceMultiplicity::History;
        return id;
    }

    void FrameGraphBuilder::setResourceSemantic(
        GraphResourceId resource,
        GraphResourceSemantic semantic)
    {
        if (resource < m_resources.size())
        {
            m_resources[resource].semantic = semantic;
        }
    }

	GraphResourceId FrameGraphBuilder::importTexture(
		ImportedResourceDesc desc)
	{
		GraphResourceId id =
			static_cast<GraphResourceId>(m_resources.size());

		m_resources.push_back({
			.id = id,
			.type = GraphResourceType::Texture,
			.ownership = ResourceOwnership::Imported,
			.imported = desc.type,
			.initialUsage = desc.initialUsage,
			.initialAccess = AccessType::Read,
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
		m_accesses.push_back({
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
		m_accesses.push_back({
			.node = node,
			.resource = resource,
			.access = AccessType::Write,
			.usage = usage
			});
	}

    void FrameGraphBuilder::readPrevious(
            GraphNodeId node,
            GraphResourceId resource,
            ResourceUsage usage)
    {
        m_accesses.push_back({
            .node = node,
            .resource = resource,
            .access = AccessType::Read,
            .usage = usage,
            .previousVersion = true
            });
    }

	void FrameGraphBuilder::clear()
	{
		m_nodes.clear();
		m_resources.clear();
		m_payloads.clear();
        m_accesses.clear();
	}

    void FrameGraphBuilder::setGraphicsPassPipeline(
        GraphNodeId node,
        PipelineId pipeline)
    {
        if (node >= m_nodes.size())
        {
            return;
        }

        const uint32_t payloadIndex =
            m_nodes[node].graphNode.payloadIndex;
        if (payloadIndex >= m_payloads.size())
        {
            return;
        }

        if (GraphicsPassData* data =
            std::get_if<GraphicsPassData>(&m_payloads[payloadIndex]))
        {
            data->pipeline = pipeline;
        }
    }

    void FrameGraphBuilder::setGraphicsPassDrawList(
        GraphNodeId node,
        DrawListKind kind)
    {
        if (node >= m_nodes.size())
        {
            return;
        }

        const uint32_t payloadIndex =
            m_nodes[node].graphNode.payloadIndex;
        if (payloadIndex >= m_payloads.size())
        {
            return;
        }

        if (GraphicsPassData* data =
            std::get_if<GraphicsPassData>(&m_payloads[payloadIndex]))
        {
            data->drawList = kind;
        }
    }

    void FrameGraphBuilder::setGraphicsPassColorLoadOp(
        GraphNodeId node,
        AttachmentLoadOp op)
    {
        if (node >= m_nodes.size())
        {
            return;
        }

        const uint32_t payloadIndex =
            m_nodes[node].graphNode.payloadIndex;
        if (payloadIndex >= m_payloads.size())
        {
            return;
        }

        if (GraphicsPassData* data =
            std::get_if<GraphicsPassData>(&m_payloads[payloadIndex]))
        {
            data->colorLoadOp = op;
        }
    }

    void FrameGraphBuilder::setGraphicsPassDepthLoadOp(
        GraphNodeId node,
        AttachmentLoadOp op)
    {
        if (node >= m_nodes.size())
        {
            return;
        }

        const uint32_t payloadIndex =
            m_nodes[node].graphNode.payloadIndex;
        if (payloadIndex >= m_payloads.size())
        {
            return;
        }

        if (GraphicsPassData* data =
            std::get_if<GraphicsPassData>(&m_payloads[payloadIndex]))
        {
            data->depthLoadOp = op;
        }
    }

    void FrameGraphBuilder::setComputePassPipeline(
        GraphNodeId node,
        PipelineId pipeline)
    {
        if (node >= m_nodes.size())
        {
            return;
        }

        const uint32_t payloadIndex =
            m_nodes[node].graphNode.payloadIndex;
        if (payloadIndex >= m_payloads.size())
        {
            return;
        }

        if (ComputePassData* data =
            std::get_if<ComputePassData>(&m_payloads[payloadIndex]))
        {
            data->pipeline = pipeline;
        }
    }

    void FrameGraphBuilder::setComputePassDispatch(
        GraphNodeId node,
        uint32_t groupCountX,
        uint32_t groupCountY,
        uint32_t groupCountZ)
    {
        if (node >= m_nodes.size())
        {
            return;
        }

        const uint32_t payloadIndex =
            m_nodes[node].graphNode.payloadIndex;
        if (payloadIndex >= m_payloads.size())
        {
            return;
        }

        if (ComputePassData* data =
            std::get_if<ComputePassData>(&m_payloads[payloadIndex]))
        {
            data->groupCountX = groupCountX;
            data->groupCountY = groupCountY;
            data->groupCountZ = groupCountZ;
        }
    }

    void FrameGraphBuilder::setTransferCopy(
        GraphNodeId node,
        GraphResourceId source,
        GraphResourceId destination)
    {
        if (node >= m_nodes.size())
        {
            return;
        }

        const uint32_t payloadIndex = m_nodes[node].graphNode.payloadIndex;
        if (payloadIndex >= m_payloads.size())
        {
            return;
        }

        if (TransferPassData* data =
            std::get_if<TransferPassData>(&m_payloads[payloadIndex]))
        {
            data->source = source;
            data->destination = destination;
        }
    }

    void FrameGraphBuilder::setNodeQueue(
        GraphNodeId node,
        QueueType queue)
    {
        if (node < m_nodes.size())
        {
            m_nodes[node].graphNode.queue = queue;
        }
    }

    void FrameGraphBuilder::setPassCadence(
        GraphNodeId node,
        PassCadence cadence)
    {
        if (node >= m_nodes.size())
        {
            return;
        }

        const uint32_t payloadIndex =
            m_nodes[node].graphNode.payloadIndex;
        if (payloadIndex >= m_payloads.size())
        {
            return;
        }

        PassPayload& payload = m_payloads[payloadIndex];
        if (GraphicsPassData* graphics =
                std::get_if<GraphicsPassData>(&payload))
        {
            graphics->execution.cadence = cadence;
            graphics->execution.invalidation = cadence == PassCadence::OnResize
                ? PassInvalidation::Resize : PassInvalidation::None;
        }
        else if (ComputePassData* compute =
                     std::get_if<ComputePassData>(&payload))
        {
            compute->execution.cadence = cadence;
            compute->execution.invalidation = cadence == PassCadence::OnResize
                ? PassInvalidation::Resize : PassInvalidation::None;
        }
        else if (TransferPassData* transfer =
                     std::get_if<TransferPassData>(&payload))
        {
            transfer->execution.cadence = cadence;
            transfer->execution.invalidation = cadence == PassCadence::OnResize
                ? PassInvalidation::Resize : PassInvalidation::None;
        }
    }

    void FrameGraphBuilder::setPassInvalidation(
        GraphNodeId node,
        PassInvalidation reasons)
    {
        if (node >= m_nodes.size())
        {
            return;
        }
        const uint32_t payloadIndex = m_nodes[node].graphNode.payloadIndex;
        if (payloadIndex >= m_payloads.size())
        {
            return;
        }

        auto set = [reasons](auto& pass)
        {
            pass.execution.cadence = PassCadence::OnInvalidation;
            pass.execution.invalidation = reasons;
        };
        PassPayload& payload = m_payloads[payloadIndex];
        if (auto* graphics = std::get_if<GraphicsPassData>(&payload)) set(*graphics);
        else if (auto* compute = std::get_if<ComputePassData>(&payload)) set(*compute);
        else if (auto* transfer = std::get_if<TransferPassData>(&payload)) set(*transfer);
    }
}
