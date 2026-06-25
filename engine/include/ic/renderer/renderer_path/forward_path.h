// ic/renderer/renderer_path/forward_path.h
#pragma once
#include "renderer_path.h"
#include "ic/renderer/frame_graph/frame_graph_types.h"
#include "ic/renderer/frame_graph/frame_graph_builder.h"

namespace ic
{
    class ForwardRendererPath : public RendererPath
    {
    public:
        GraphNodeId geometryNode;
        GraphNodeId shadowNode;
        GraphNodeId lightingNode;
        GraphNodeId postNode;

        GraphResourceId gbuffer;
        GraphResourceId shadowMap;
        GraphResourceId litColor;

        void buildFrameGraph(FrameGraphBuilder& builder) override
        {
            // Resources
            gbuffer = builder.createTexture();
            shadowMap = builder.createTexture();
            litColor = builder.createTexture();

            // Pass 0: Geometry
            geometryNode = builder.addGraphNode(
                GeometryPassData{},
                GraphNodeType::Graphics,
                QueueType::Graphics);

            builder.write(
                geometryNode,
                gbuffer,
                ResourceUsage::ColorAttachment);


            // Pass 1: Shadow
            shadowNode = builder.addGraphNode(
                ShadowPassData{},
                GraphNodeType::Graphics,
                QueueType::Graphics);

            builder.write(
                shadowNode,
                shadowMap,
                ResourceUsage::DepthAttachment);


            // Pass 2: Lighting
            lightingNode = builder.addGraphNode(
                LightingPassData{},
                GraphNodeType::Graphics,
                QueueType::Graphics);

            builder.read(
                lightingNode,
                gbuffer,
                ResourceUsage::SampledTexture);

            builder.read(
                lightingNode,
                shadowMap,
                ResourceUsage::SampledTexture);

            builder.write(
                lightingNode,
                litColor,
                ResourceUsage::ColorAttachment);

            // Pass 3: Post
            postNode = builder.addGraphNode(
                PostProcessPassData{},
                GraphNodeType::Graphics,
                QueueType::Graphics);

            builder.read(
                postNode,
                litColor,
                ResourceUsage::SampledTexture);
        }
    };

}