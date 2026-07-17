#include "ic/renderer/vulkan_backend/vulkan_pass_recorders.h"

#include "ic/renderer/renderer_gpu_assets.h"

#include <algorithm>
#include <iterator>
#include <vector>

namespace ic
{
    void recordTransferCopy(
        VkCommandBuffer cmd,
        const VulkanTransferCopy& copy)
    {
        if (copy.isBuffer)
        {
            VkBufferCopy region{};
            region.size = copy.bufferSize;
            vkCmdCopyBuffer(
                cmd,
                copy.sourceBuffer,
                copy.destinationBuffer,
                1,
                &region);
            return;
        }

        VkImageCopy region{};
        region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.srcSubresource.layerCount = 1;
        region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.dstSubresource.layerCount = 1;
        region.extent = { copy.width, copy.height, 1 };
        vkCmdCopyImage(
            cmd,
            copy.sourceImage,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            copy.destinationImage,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &region);
    }

    void recordComputeStorageBufferTest(
        const VulkanPassContext& ctx,
        const VulkanComputeStorageBufferTest& test)
    {
        VkCommandBuffer cmd = ctx.cmd;
        vkCmdBindDescriptorSets(
            cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            test.pipelineLayout,
            0,
            1,
            &test.descriptorSet,
            0,
            nullptr);
        vkCmdDispatch(
            cmd, test.groupCountX, test.groupCountY, test.groupCountZ);

        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = test.buffer;
        barrier.offset = 0;
        barrier.size = test.bufferSize;

        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0, nullptr,
            1, &barrier,
            0, nullptr);
    }

    bool recordHiZPyramid(
        const VulkanPassContext& ctx,
        const VulkanComputePipeline& pipeline,
        const VulkanHiZInputs& inputs)
    {
        VulkanGraphResourceEntry* sceneDepth =
            ctx.resources->entry(inputs.sceneDepthId);
        VulkanGraphResourceEntry* hiZ = ctx.resources->entry(inputs.hiZId);
        if (!sceneDepth || !hiZ || hiZ->mipViews.empty())
        {
            return false;
        }

        if (inputs.hiZDebugResourceOut)
        {
            *inputs.hiZDebugResourceOut = inputs.hiZId;
        }

        if (inputs.frameConstants == VK_NULL_HANDLE || !inputs.hiZPool)
        {
            return false;
        }

        VkCommandBuffer cmd = ctx.cmd;

        if (hiZ->layout != VK_IMAGE_LAYOUT_GENERAL)
        {
            VkImageMemoryBarrier hiZWriteBarrier{};
            hiZWriteBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            hiZWriteBarrier.oldLayout = hiZ->layout;
            hiZWriteBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            hiZWriteBarrier.srcAccessMask =
                hiZ->layout == VK_IMAGE_LAYOUT_UNDEFINED
                    ? 0
                    : VK_ACCESS_SHADER_READ_BIT;
            hiZWriteBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            hiZWriteBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            hiZWriteBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            hiZWriteBarrier.image = hiZ->texture.image;
            hiZWriteBarrier.subresourceRange.aspectMask =
                VK_IMAGE_ASPECT_COLOR_BIT;
            hiZWriteBarrier.subresourceRange.levelCount = hiZ->mipLevels;
            hiZWriteBarrier.subresourceRange.layerCount = 1;

            vkCmdPipelineBarrier(
                cmd,
                hiZ->layout == VK_IMAGE_LAYOUT_UNDEFINED
                    ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                    : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr,
                0, nullptr,
                1,
                &hiZWriteBarrier);
            hiZ->layout = VK_IMAGE_LAYOUT_GENERAL;
        }

        VkDescriptorPoolSize poolSizes[3]{};
        constexpr uint32_t kMaxHiZDescriptorSets = 32;
        poolSizes[0] =
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kMaxHiZDescriptorSets };
        poolSizes[1] =
            { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kMaxHiZDescriptorSets };
        poolSizes[2] =
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kMaxHiZDescriptorSets };

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = kMaxHiZDescriptorSets;
        poolInfo.poolSizeCount = static_cast<uint32_t>(std::size(poolSizes));
        poolInfo.pPoolSizes = poolSizes;

        if (*inputs.hiZPool == VK_NULL_HANDLE &&
            vkCreateDescriptorPool(
                ctx.device,
                &poolInfo,
                nullptr,
                inputs.hiZPool) != VK_SUCCESS)
        {
            return false;
        }

        std::vector<VkDescriptorSetLayout> layouts(
            hiZ->mipLevels,
            pipeline.descriptorSetLayout);
        std::vector<VkDescriptorSet> sets(hiZ->mipLevels);

        VkDescriptorSetAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocateInfo.descriptorPool = *inputs.hiZPool;
        allocateInfo.descriptorSetCount = hiZ->mipLevels;
        allocateInfo.pSetLayouts = layouts.data();
        if (vkAllocateDescriptorSets(
                ctx.device,
                &allocateInfo,
                sets.data()) != VK_SUCCESS)
        {
            return false;
        }

        std::vector<VkDescriptorBufferInfo> frameInfos(hiZ->mipLevels);
        std::vector<VkDescriptorImageInfo> sourceInfos(hiZ->mipLevels);
        std::vector<VkDescriptorImageInfo> outputInfos(hiZ->mipLevels);
        std::vector<VkWriteDescriptorSet> writes;
        writes.reserve(hiZ->mipLevels * 3u);

        for (uint32_t mip = 0; mip < hiZ->mipLevels; ++mip)
        {
            frameInfos[mip].buffer = inputs.frameConstants;
            frameInfos[mip].range = sizeof(GpuFrameData);

            sourceInfos[mip].imageLayout =
                mip == 0
                    ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                    : VK_IMAGE_LAYOUT_GENERAL;
            sourceInfos[mip].imageView =
                mip == 0 ? sceneDepth->view : hiZ->mipViews[mip - 1u];

            outputInfos[mip].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            outputInfos[mip].imageView = hiZ->mipViews[mip];

            VkWriteDescriptorSet frameWrite{};
            frameWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            frameWrite.dstSet = sets[mip];
            frameWrite.dstBinding = 0;
            frameWrite.descriptorCount = 1;
            frameWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            frameWrite.pBufferInfo = &frameInfos[mip];
            writes.push_back(frameWrite);

            VkWriteDescriptorSet sourceWrite{};
            sourceWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            sourceWrite.dstSet = sets[mip];
            sourceWrite.dstBinding = 20;
            sourceWrite.descriptorCount = 1;
            sourceWrite.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            sourceWrite.pImageInfo = &sourceInfos[mip];
            writes.push_back(sourceWrite);

            VkWriteDescriptorSet outputWrite{};
            outputWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            outputWrite.dstSet = sets[mip];
            outputWrite.dstBinding = 21;
            outputWrite.descriptorCount = 1;
            outputWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            outputWrite.pImageInfo = &outputInfos[mip];
            writes.push_back(outputWrite);
        }

        vkUpdateDescriptorSets(
            ctx.device,
            static_cast<uint32_t>(writes.size()),
            writes.data(),
            0,
            nullptr);

        vkCmdBindPipeline(
            cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.pipeline);

        uint32_t mipWidth = hiZ->width;
        uint32_t mipHeight = hiZ->height;
        for (uint32_t mip = 0; mip < hiZ->mipLevels; ++mip)
        {
            vkCmdBindDescriptorSets(
                cmd,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline.pipelineLayout,
                0,
                1,
                &sets[mip],
                0,
                nullptr);

            vkCmdDispatch(
                cmd,
                (mipWidth + 7u) / 8u,
                (mipHeight + 7u) / 8u,
                1);

            VkImageMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask =
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = hiZ->texture.image;
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.baseMipLevel = mip;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.layerCount = 1;

            vkCmdPipelineBarrier(
                cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr,
                0, nullptr,
                1, &barrier);

            mipWidth = std::max(1u, mipWidth >> 1u);
            mipHeight = std::max(1u, mipHeight >> 1u);
        }

        return true;
    }

    bool recordGpuFrustumCull(
        const VulkanPassContext& ctx,
        const VulkanComputePipeline& pipeline,
        const VulkanCullBuffers& buffers)
    {
        VkDescriptorPoolSize poolSizes[3]{};
        constexpr uint32_t kMaxGpuCullDescriptorSets = 4;
        poolSizes[0] =
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kMaxGpuCullDescriptorSets };
        poolSizes[1] =
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kMaxGpuCullDescriptorSets * 9u };
        poolSizes[2] =
            { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kMaxGpuCullDescriptorSets };

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = kMaxGpuCullDescriptorSets;
        poolInfo.poolSizeCount = static_cast<uint32_t>(std::size(poolSizes));
        poolInfo.pPoolSizes = poolSizes;

        if (*buffers.gpuCullPool == VK_NULL_HANDLE &&
            vkCreateDescriptorPool(
                ctx.device,
                &poolInfo,
                nullptr,
                buffers.gpuCullPool) != VK_SUCCESS)
        {
            return false;
        }

        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        VkDescriptorSetAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocateInfo.descriptorPool = *buffers.gpuCullPool;
        allocateInfo.descriptorSetCount = 1;
        allocateInfo.pSetLayouts = &pipeline.descriptorSetLayout;
        if (vkAllocateDescriptorSets(
                ctx.device,
                &allocateInfo,
                &descriptorSet) != VK_SUCCESS)
        {
            return false;
        }

        VkDescriptorBufferInfo frameInfo{};
        frameInfo.buffer = buffers.frameConstants;
        frameInfo.range = sizeof(GpuFrameData);

        VkDescriptorBufferInfo boundsInfo{};
        boundsInfo.buffer = buffers.instanceBounds;
        boundsInfo.range = buffers.instanceBoundsSize;

        VkDescriptorBufferInfo visibleInfo{};
        visibleInfo.buffer = buffers.visibleInstances;
        visibleInfo.range = buffers.visibleInstancesSize;

        VkDescriptorBufferInfo countInfo{};
        countInfo.buffer = buffers.visibleInstanceCount;
        countInfo.range = buffers.visibleInstanceCountSize;

        VkDescriptorBufferInfo drawInputInfo{};
        drawInputInfo.buffer = buffers.drawInputs;
        drawInputInfo.range = buffers.drawInputsSize;
        VkDescriptorBufferInfo indirectInfo{};
        indirectInfo.buffer = buffers.indirectArguments;
        indirectInfo.range = buffers.indirectArgumentsSize;
        VkDescriptorBufferInfo metadataInfo{};
        metadataInfo.buffer = buffers.drawMetadata;
        metadataInfo.range = buffers.drawMetadataSize;
        VkDescriptorBufferInfo binCountInfo{};
        binCountInfo.buffer = buffers.binCounts;
        binCountInfo.range = buffers.binCountsSize;
        VkDescriptorBufferInfo classificationInfo{};
        classificationInfo.buffer = buffers.cullClassification;
        classificationInfo.range = buffers.cullClassificationSize;
        VkDescriptorBufferInfo statsInfo{};
        statsInfo.buffer = buffers.cullStats;
        statsInfo.range = buffers.cullStatsSize;
        VkDescriptorImageInfo previousHiZInfo{};
        previousHiZInfo.imageView = buffers.previousHiZ;
        previousHiZInfo.imageLayout = buffers.previousHiZLayout;

        VkDescriptorBufferInfo* infos[] =
        {
            &frameInfo,
            &boundsInfo,
            &visibleInfo,
            &countInfo,
            &drawInputInfo,
            &indirectInfo,
            &metadataInfo,
            &binCountInfo,
            &statsInfo,
            &classificationInfo
        };

        VkWriteDescriptorSet writes[11]{};
        const uint32_t bindings[] =
            { 0u, 22u, 23u, 24u, 25u, 26u, 27u, 28u, 30u, 31u, 29u };
        const VkDescriptorType types[] =
        {
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
        };
        for (uint32_t i = 0; i < 10u; ++i)
        {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptorSet;
            writes[i].dstBinding = bindings[i];
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = types[i];
            writes[i].pBufferInfo = infos[i];
        }
        uint32_t writeCount = 10u;
        if (buffers.previousHiZ != VK_NULL_HANDLE)
        {
            writes[10].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[10].dstSet = descriptorSet;
            writes[10].dstBinding = bindings[10];
            writes[10].descriptorCount = 1;
            writes[10].descriptorType = types[10];
            writes[10].pImageInfo = &previousHiZInfo;
            writeCount = 11u;
        }
        vkUpdateDescriptorSets(
            ctx.device, writeCount, writes, 0, nullptr);

        vkCmdBindDescriptorSets(
            ctx.cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.pipelineLayout,
            0,
            1,
            &descriptorSet,
            0,
            nullptr);

        return true;
    }

    bool recordGpuOcclusionValidation(
        const VulkanPassContext& ctx,
        const VulkanComputePipeline& pipeline,
        const VulkanOcclusionValidationInputs& inputs)
    {
        if (!inputs.descriptorPool ||
            inputs.currentHiZ == VK_NULL_HANDLE)
        {
            return false;
        }

        VkDescriptorSet set = VK_NULL_HANDLE;
        VkDescriptorSetAllocateInfo allocate{};
        allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate.descriptorPool = *inputs.descriptorPool;
        allocate.descriptorSetCount = 1;
        allocate.pSetLayouts = &pipeline.descriptorSetLayout;
        if (vkAllocateDescriptorSets(
                ctx.device, &allocate, &set) != VK_SUCCESS)
        {
            return false;
        }

        VkDescriptorBufferInfo frame{
            inputs.frameConstants, 0, sizeof(GpuFrameData) };
        VkDescriptorBufferInfo bounds{
            inputs.instanceBounds, 0, inputs.instanceBoundsSize };
        VkDescriptorBufferInfo stats{
            inputs.cullStats, 0, inputs.cullStatsSize };
        VkDescriptorBufferInfo classification{
            inputs.cullClassification, 0,
            inputs.cullClassificationSize };
        VkDescriptorImageInfo hiZ{};
        hiZ.imageView = inputs.currentHiZ;
        hiZ.imageLayout = inputs.currentHiZLayout;

        VkWriteDescriptorSet writes[5]{};
        const uint32_t bindings[] = { 0u, 22u, 29u, 30u, 31u };
        const VkDescriptorType types[] = {
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
        VkDescriptorBufferInfo* bufferInfos[] = {
            &frame, &bounds, nullptr, &stats, &classification };
        for (uint32_t i = 0; i < 5u; ++i)
        {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = bindings[i];
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = types[i];
            if (i == 2u)
            {
                writes[i].pImageInfo = &hiZ;
            }
            else
            {
                writes[i].pBufferInfo = bufferInfos[i];
            }
        }
        vkUpdateDescriptorSets(ctx.device, 5, writes, 0, nullptr);
        vkCmdBindDescriptorSets(
            ctx.cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.pipelineLayout,
            0, 1, &set, 0, nullptr);
        return true;
    }

    void recordSceneGeometryDraws(
        VkCommandBuffer cmd,
        VkPipelineLayout pipelineLayout,
        std::span<const VulkanGpuScene::DrawItem> draws,
        std::span<const VulkanGpuScene::GeometryBin> geometryBins,
        bool useGpuDriven,
        const VulkanIndirectDrawStream& indirectStream,
        const VulkanResolveNativeModelFn& resolveNativeModel)
    {
        if (useGpuDriven)
        {
            const DrawConstants unusedFallbackConstants{};
            vkCmdPushConstants(
                cmd,
                pipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(unusedFallbackConstants),
                &unusedFallbackConstants);
            for (uint32_t binIndex = 0;
                 binIndex < geometryBins.size();
                 ++binIndex)
            {
                const VulkanGpuScene::GeometryBin& bin = geometryBins[binIndex];
                if (!bin.model || bin.maxDrawCount == 0u)
                {
                    continue;
                }
                VulkanUploadedModel* model = resolveNativeModel(bin.model);
                if (!model)
                {
                    continue;
                }
                VkDeviceSize vertexOffset = 0;
                vkCmdBindVertexBuffers(cmd, 0, 1,
                    &model->vertexBuffer.buffer, &vertexOffset);
                vkCmdBindIndexBuffer(cmd, model->indexBuffer.buffer,
                    0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexedIndirectCount(
                    cmd,
                    indirectStream.indirectArguments,
                    static_cast<VkDeviceSize>(bin.commandOffset) *
                        sizeof(GpuIndexedIndirectArguments),
                    indirectStream.binCounts,
                    static_cast<VkDeviceSize>(binIndex) * sizeof(uint32_t),
                    bin.maxDrawCount,
                    sizeof(GpuIndexedIndirectArguments));
            }
            return;
        }

        bool haveBoundModel = false;
        AssetHandle boundModelHandle{};
        VulkanUploadedModel* boundModel = nullptr;
        for (const VulkanGpuScene::DrawItem& draw : draws)
        {
            if (!haveBoundModel || draw.model != boundModelHandle)
            {
                boundModel = resolveNativeModel(draw.model);
                boundModelHandle = draw.model;
                haveBoundModel = true;
                if (!boundModel)
                {
                    continue;
                }

                VkDeviceSize offset = 0;
                vkCmdBindVertexBuffers(
                    cmd,
                    0,
                    1,
                    &boundModel->vertexBuffer.buffer,
                    &offset);
                vkCmdBindIndexBuffer(
                    cmd,
                    boundModel->indexBuffer.buffer,
                    0,
                    VK_INDEX_TYPE_UINT32);
            }

            if (!boundModel || draw.meshIndex >= boundModel->meshes.size())
            {
                continue;
            }

            const GpuMesh& mesh = boundModel->meshes[draw.meshIndex];
            if (mesh.indexCount == 0)
            {
                continue;
            }

            DrawConstants constants{};
            constants.objectIndex = draw.objectIndex;
            constants.meshIndex = draw.meshIndex;
            constants.materialIndex = draw.materialIndex;

            vkCmdPushConstants(
                cmd,
                pipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(constants),
                &constants);

            vkCmdDrawIndexed(
                cmd,
                mesh.indexCount,
                1,
                mesh.firstIndex,
                0,
                0);
        }
    }
}
