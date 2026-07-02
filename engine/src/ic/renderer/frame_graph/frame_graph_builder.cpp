#include "ic/common/ic_pch.h"
#include "ic/renderer/frame_graph/frame_graph_builder.h"
#include "ic/core/memory/frame_arena.h"
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
        FrameGraphBuilder::addComputePass(std::string_view name)
    {
        ComputePassData data{};
        data.name = name;

        const GraphNodeId node =
            addGraphNode(
                data,
                GraphNodeType::Compute,
                QueueType::Compute);

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
            .initialUsage = ResourceUsage::StorageBuffer,
            .initialAccess = AccessType::Write,
            .bufferDesc = desc,
			.firstAccess = 0,
			.accessCount = 0
			});

		return id;
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
}
