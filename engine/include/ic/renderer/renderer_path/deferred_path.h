#pragma once


namespace ic
{
    // EXAMPLE PATH
    // 
    // 
    //class DeferredPath : public RenderPath
    //{
    //public:
    //    void setupGraph(FrameGraphBuilder& builder, const RenderScenePacket& packet) override
    //    {
    //        builder.addNode("GPU_Culling", QueueType::Compute)
    //            .write(packet.hzbTexture) // Dependency tracking
    //            .execute([=](const RenderContext& ctx) {
    //            // Compute passes use raw backend dispatches directly
    //            ctx.backend->bindComputePipeline(m_cullPipeline);
    //            ctx.backend->dispatchCompute(packet.objectCount / 64, 1, 1);
    //                });

    //        builder.addNode("GBuffer_Pass", QueueType::Graphics)
    //            .read(packet.hzbTexture)
    //            .write(packet.gBufferAlbedo)
    //            .execute([=](const RenderContext& ctx) {
    //            // Raster passes handle their own drawing commands
    //            ctx.backend->beginRenderPass(m_gBufferRenderTargets);
    //            ctx.backend->bindGraphicsPipeline(m_gBufferPipeline);
    //            ctx.backend->dispatchIndirectDraws(packet.drawCommandBuffer, packet.drawCount);
    //            ctx.backend->endRenderPass();
    //                });

    //        builder.addPass("Deferred_Lighting", QueueType::Compute)
    //            .read(packet.gBufferAlbedo)
    //            .write(packet.hdrColorTarget)
    //            .execute([=](const RenderContext& ctx) {
    //            ctx.backend->bindComputePipeline(m_lightingPipeline);
    //            ctx.backend->dispatchCompute(viewportWidth / 16, viewportHeight / 16, 1);
    //                });
    //    }
    //private:
    //    PipelineHandle m_cullPipeline;
    //    PipelineHandle m_gBufferPipeline;
    //    PipelineHandle m_lightingPipeline;
    //    // Target attachments etc...
    //};
}