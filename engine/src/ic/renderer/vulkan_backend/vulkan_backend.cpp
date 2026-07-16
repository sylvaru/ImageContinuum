#include "ic/renderer/vulkan_backend/vulkan_backend.h"
#include "ic/core/app_base.h"
#include "ic/core/asset_manager.h"
#include "ic/renderer/pipeline_library.h"
#include "ic/renderer/gpu_driven_submission.h"
#include "ic/renderer/vulkan_backend/vulkan_pass_recorders.h"
#include "ic/renderer/vulkan_backend/vulkan_offscreen_pass_recorder.h"
#include "ic/renderer/vulkan_backend/vulkan_scene_pass_recorder.h"
#include "ic/renderer/renderer_specification.h"
#include "ic/renderer/path_tracing/path_trace_scene_builder.h"
#include "ic/renderer/renderer_common/renderer_util.h"
#include "ic/interface/window.h"
#include "ic/core/frame_context.h"
#include "ic/renderer/frame_graph/frame_graph_executor.h"
#include "ic/scene/scene_render_view.h"

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <spdlog/spdlog.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace ic
{
    namespace
    {
        void throwIfFailed(VkResult result, const char* message)
        {
            if (result != VK_SUCCESS)
            {
                throw std::runtime_error(message);
            }
        }

        uint32_t mipCountForSize(uint32_t size)
        {
            uint32_t levels = 1;
            while (size > 1)
            {
                size >>= 1u;
                ++levels;
            }
            return levels;
        }

        void transitionImage(
            VkCommandBuffer cmd,
            VkImage image,
            VkImageLayout oldLayout,
            VkImageLayout newLayout,
            VkAccessFlags srcAccess,
            VkAccessFlags dstAccess,
            VkPipelineStageFlags srcStage,
            VkPipelineStageFlags dstStage,
            uint32_t baseMipLevel = 0,
            uint32_t levelCount = 1)
        {
            VkImageMemoryBarrier barrier{
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
            barrier.oldLayout = oldLayout;
            barrier.newLayout = newLayout;
            barrier.srcAccessMask = srcAccess;
            barrier.dstAccessMask = dstAccess;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image;
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.baseMipLevel = baseMipLevel;
            barrier.subresourceRange.levelCount = levelCount;
            barrier.subresourceRange.layerCount = 1;
            vkCmdPipelineBarrier(
                cmd, srcStage, dstStage, 0,
                0, nullptr, 0, nullptr, 1, &barrier);
        }
    }

	void VulkanBackend::init(
		const RendererSpecification& spec,
        const PipelineLibrary& pipelineLibrary,
		Window& window,
		uint32_t workerCount)
	{
		m_instance.init();

		m_platform.init(
			m_instance.instance(),
			window);

		m_adapter.init(
			m_instance.instance(),
			m_platform.surface());

		m_device.init(
			m_adapter,
			m_platform.surface());

        m_resourceAllocator.init(
            m_instance.instance(),
            m_adapter.device(),
            m_device.device(),
            m_device.info());

        m_swapchain.setVsyncEnabled(spec.settings.vsync);

		m_swapchain.init(
			m_adapter,
			m_device,
			m_platform.surface(),
			window);
        // Seed the authoritative framebuffer size so the first reconcile does
        // not spuriously recreate a just-created swapchain.
        m_swapchain.windowFramebufferSize(
            m_lastFramebufferWidth, m_lastFramebufferHeight);

        const uint32_t framesInFlight =
            spec.framesInFlight == 0 ? 1 : spec.framesInFlight;

        const uint32_t workerSlots =
            workerCount == 0 ? 1 : workerCount;
        m_workerSlots = workerSlots;

        m_frameExecutor.init(m_device, m_swapchain, framesInFlight);

		m_commandSystem.init(
			m_device.device(),
			m_adapter.info().queueFamilies,
			framesInFlight,
			workerSlots);

        m_descriptorSystem.init(
            m_device.device(),
            m_device.info());

        m_graphResourceRegistry.init(
            m_device,
            m_resourceAllocator,
            framesInFlight);
        // The graph-owned GPU-driven indirect-argument buffer is sized in native
        // Vulkan command units (VkDrawIndexedIndirectCommand).
        m_graphResourceRegistry.setNativeIndirectCommandStride(
            sizeof(VkDrawIndexedIndirectCommand));

        m_retirementQueue.init(
            m_device.device(), m_resourceAllocator, framesInFlight);

        m_uploadScheduler.init(
            m_device, framesInFlight);

        m_gpuScene.init(m_resourceAllocator, framesInFlight);

        m_pipelineManager.init(m_device.device());

        m_pipelineLibrary = &pipelineLibrary;

        if (spec.useDebugGui)
        {
            initImGui(window);
        }
	}

    void VulkanBackend::shutdown()
    {
        if (m_device.device() != VK_NULL_HANDLE)
        {
            vkDeviceWaitIdle(m_device.device());
        }

        shutdownImGui();
        destroySceneResources();
        destroyEnvironmentResources();
        destroyClusteredForwardResources();
        destroyPathTraceResources();
        m_uploadScheduler.shutdown();
        m_retirementQueue.drain();
        m_graphResourceRegistry.shutdown();
        m_pipelineManager.shutdown();

        m_frameExecutor.shutdown();

        m_descriptorSystem.shutdown();
        m_resourceAllocator.shutdown();
        m_commandSystem.shutdown();
        m_swapchain.shutdown();
        m_device.shutdown();
        m_platform.shutdown();
        m_instance.shutdown();

        spdlog::info("[VulkanBackend] Shutdown");
    }

    RenderSurfaceState VulkanBackend::reconcileRenderSurface()
    {
        RenderSurfaceState surface{};

        uint32_t framebufferWidth = 0;
        uint32_t framebufferHeight = 0;
        m_swapchain.windowFramebufferSize(framebufferWidth, framebufferHeight);

        // Minimized / zero drawable area: leave the swapchain untouched and
        // report the frame as non-renderable so the renderer skips it entirely.
        if (framebufferWidth == 0 || framebufferHeight == 0)
        {
            return surface;
        }

        // Recreate when the framebuffer changed or the swapchain is no longer
        // valid (e.g. a prior out-of-date acquire/present). onSwapchainRecreated
        // bumps the generation so the renderer rebuilds the graph to the new
        // extent and re-materializes every resource before this frame records.
        if (framebufferWidth != m_lastFramebufferWidth ||
            framebufferHeight != m_lastFramebufferHeight ||
            !m_swapchain.validForRendering())
        {
            onSwapchainRecreated();
            m_lastFramebufferWidth = framebufferWidth;
            m_lastFramebufferHeight = framebufferHeight;
        }

        if (!m_swapchain.validForRendering())
        {
            return surface;
        }

        // Report the swapchain's actual (surface-clamped) extent, which is what
        // the images physically are and therefore what the graph must match.
        const VkExtent2D extent = m_swapchain.extent();
        surface.width = extent.width;
        surface.height = extent.height;
        surface.generation = m_swapchainGeneration;
        surface.renderable = true;
        return surface;
    }

    bool VulkanBackend::execute(
        const CompiledGraphPlan& plan,
        const GraphExecutionContext& execution,
        const FrameContext& ctx,
        const SceneRenderView& scene)
    {
        if (!m_frameExecutor.ready())
        {
            return false;
        }

        const uint32_t frameSlot =
            static_cast<uint32_t>(
                ctx.frameIndex % m_frameExecutor.framesInFlight());

        m_frameExecutor.waitForFrameSlot(frameSlot);
        m_retirementQueue.beginFrame(frameSlot);
        m_uploadScheduler.beginFrame(frameSlot);

        if (frameSlot < m_gpuScene.frameSlotCount())
        {
            VulkanGpuSceneFrameResources& resources =
                m_gpuScene.frameResources(frameSlot);
            if (resources.hiZDescriptorPool != VK_NULL_HANDLE)
            {
                vkResetDescriptorPool(
                    m_device.device(),
                    resources.hiZDescriptorPool,
                    0);
            }
            if (resources.gpuCullDescriptorPool != VK_NULL_HANDLE)
            {
                vkResetDescriptorPool(
                    m_device.device(),
                    resources.gpuCullDescriptorPool,
                    0);
            }
        }

        const VulkanFrameExecutor::AcquiredFrame frame =
            m_frameExecutor.acquire(frameSlot);
        if (!frame.acquired)
        {
            onSwapchainRecreated();
            return false;
        }

        VkImage swapchainImage = frame.swapchainImage;

        // The slot's prior GPU work completed in waitForFrameSlot above, so any
        // resources retired the last time this slot was used are now safe to
        // free without adding a GPU wait.
        m_graphResourceRegistry.recycleFrameSlot(frameSlot);

        VulkanGraphResourceImports graphImports{};
        graphImports.swapchainImage = swapchainImage;
        const VkExtent2D graphExtent = m_swapchain.extent();
        m_graphResourceRegistry.materialize(
            plan,
            frameSlot,
            graphExtent.width,
            graphExtent.height,
            graphImports);

        m_commandSystem.beginFrame(frameSlot);

        std::vector<VkCommandBuffer> commandBuffers;
        executeGraph(
            plan,
            execution,
            ctx,
            scene,
            swapchainImage,
            frame.initialLayout,
            commandBuffers);
        recordImGui(
            ctx,
            swapchainImage,
            commandBuffers);

        const VulkanUploadDependency uploadDependency =
            m_uploadScheduler.flush();
        const bool presented = m_frameExecutor.submitAndPresent(
            plan, commandBuffers, frameSlot, execution, uploadDependency);

        if (!presented)
        {
            onSwapchainRecreated();
        }
        return presented;
    }


    void VulkanBackend::executeGraph(
        const CompiledGraphPlan& plan,
        const GraphExecutionContext& execution,
        const FrameContext& ctx,
        const SceneRenderView& scene,
        VkImage swapchainImage,
        VkImageLayout swapchainInitialLayout,
        std::vector<VkCommandBuffer>& commandBuffers)
    {
        // Resolve lazy pipeline/resource state before worker threads begin.
        // Recording jobs may only read this shared renderer state.
        ensureDepthTarget();
        ensureClusteredForwardResources();
        // Resolve the graph-owned GPU-driven + cluster buffer handles before any
        // pass (prepareSceneResources -> updateFrameDescriptors binds them).
        resolveGpuDrivenBuffers(plan);
        resolveClusterBuffers(plan);
        for (const ExecutionNode& node : plan.nodes)
        {
            if (node.type == GraphNodeType::Graphics)
            {
                const GraphicsPipelineHandle handle =
                    pipelineForNode(plan, node);
                const GraphicsPassData* pass =
                    node.payloadIndex < plan.payloads.size()
                        ? std::get_if<GraphicsPassData>(
                            &plan.payloads[node.payloadIndex])
                        : nullptr;
                if (pass && pass->drawList == DrawListKind::SceneGeometry)
                {
                    (void)prepareSceneResources(ctx, scene, handle);
                }
            }
            else if (node.type == GraphNodeType::Compute)
            {
                (void)computePipelineForNode(plan, node);
            }
        }

        // Prepared scene data now exists; push the cull inputs into the
        // graph-owned per-frame-slot buffers before any worker records the cull.
        uploadGpuDrivenInputs();

        auto recordNode =
            [&](GraphNodeId nodeId, uint32_t workerIndex)
            {
                auto lease =
                    m_commandSystem.acquireFrameCommandBuffer(
                        static_cast<uint32_t>(ctx.frameIndex % m_frameExecutor.framesInFlight()),
                        workerIndex,
                        plan.nodes[nodeId].queue);

                VkCommandBuffer cmd =
                    lease.commandBuffer();

                VkCommandBufferBeginInfo beginInfo{};
                beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                beginInfo.pInheritanceInfo = nullptr;

                if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS)
                {
                    throw std::runtime_error(
                        "Failed to begin Vulkan command buffer.");
                }

                const ExecutionNode& node =
                    plan.nodes[nodeId];

                applyBarriers(
                    cmd,
                    plan,
                    node,
                    swapchainImage,
                    swapchainInitialLayout);

                if (execution.shouldExecute(node.nodeId))
                {
                    dispatchNode(
                        plan,
                        node,
                        ctx,
                        scene,
                        cmd,
                        swapchainImage);
                }

                if (node.nodeId < plan.nodeSchedules.size())
                {
                    const NodeSchedule& schedule = plan.nodeSchedules[node.nodeId];
                    for (uint32_t i = 0; i < schedule.outgoingBarrierCount; ++i)
                    {
                        const ResourceBarrier& barrier =
                            plan.barriers[plan.outgoingBarrierIndices[
                                schedule.firstOutgoingBarrier + i]];
                        if (barrier.fromNode != barrier.toNode &&
                            plan.nodes[barrier.fromNode].queue !=
                                plan.nodes[barrier.toNode].queue)
                        {
                            recordBarrier(
                                cmd, barrier, plan.resources, swapchainImage,
                                swapchainInitialLayout,
                                plan.nodes[barrier.fromNode].queue,
                                plan.nodes[barrier.toNode].queue,
                                true, false, node.queue);
                        }
                    }
                }

                if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
                {
                    throw std::runtime_error(
                        "Failed to end Vulkan command buffer.");
                }

                return cmd;
            };

        recordFrameGraph(
            plan,
            ctx.services ? ctx.services->jobSystem : nullptr,
            m_workerSlots,
            recordNode,
            commandBuffers);
    }

    void VulkanBackend::applyBarriers(
        VkCommandBuffer cmd,
        const CompiledGraphPlan& plan,
        const ExecutionNode& node,
        VkImage swapchainImage,
        VkImageLayout swapchainInitialLayout)
    {
        if (node.nodeId >= plan.nodeSchedules.size())
        {
            for (const ResourceBarrier& barrier : plan.barriers)
            {
                if (barrier.toNode != node.nodeId)
                {
                    continue;
                }

                recordBarrier(
                    cmd,
                    barrier,
                    plan.resources,
                    swapchainImage,
                    swapchainInitialLayout,
                    plan.nodes[barrier.fromNode].queue,
                    plan.nodes[barrier.toNode].queue,
                    false,
                    barrier.fromNode != barrier.toNode &&
                    plan.nodes[barrier.fromNode].queue !=
                        plan.nodes[barrier.toNode].queue,
                    node.queue);
            }

            return;
        }

        const NodeSchedule& schedule =
            plan.nodeSchedules[node.nodeId];

        for (uint32_t i = 0; i < schedule.incomingBarrierCount; ++i)
        {
            const uint32_t barrierIndex =
                plan.incomingBarrierIndices[
                    schedule.firstIncomingBarrier + i];
            if (barrierIndex >= plan.barriers.size())
            {
                continue;
            }

            const ResourceBarrier& barrier = plan.barriers[barrierIndex];
            recordBarrier(
                cmd,
                barrier,
                plan.resources,
                swapchainImage,
                swapchainInitialLayout,
                plan.nodes[barrier.fromNode].queue,
                plan.nodes[barrier.toNode].queue,
                false,
                barrier.fromNode != barrier.toNode &&
                plan.nodes[barrier.fromNode].queue !=
                    plan.nodes[barrier.toNode].queue,
                node.queue);
        }
    }

    void VulkanBackend::recordBarrier(
        VkCommandBuffer cmd,
        const ResourceBarrier& barrier,
        std::span<const GraphResource> resources,
        VkImage swapchainImage,
        VkImageLayout swapchainInitialLayout,
        QueueType sourceQueue,
        QueueType destinationQueue,
        bool crossQueueRelease,
        bool crossQueueAcquire,
        QueueType commandQueue)
    {
        (void)sourceQueue;
        (void)destinationQueue;
        // Graph resources use concurrent sharing across the selected queue
        // families. The source-side barrier performs any layout transition;
        // the timeline-semaphore wait supplies the destination visibility.
        // Recording a second acquire transition would repeat oldLayout ->
        // newLayout after the release already completed it.
        if (crossQueueAcquire)
        {
            return;
        }

        const auto nativeQueue = [this](QueueType queue)
        {
            switch (queue)
            {
            case QueueType::Graphics: return m_device.graphicsQueue();
            case QueueType::Compute: return m_device.computeQueue();
            case QueueType::Transfer: return m_device.transferQueue();
            }
            return m_device.graphicsQueue();
        };
        const bool releaseOnly = crossQueueRelease &&
            nativeQueue(sourceQueue) != nativeQueue(destinationQueue);

        VkImageMemoryBarrier vkBarrier{};
        vkBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;

        const GraphResource& resource =
            resources[barrier.resource];

        if (resource.type == GraphResourceType::Buffer)
        {
            // A previous-version barrier targets the previous frame's instance.
            const VulkanGraphResourceEntry* entry =
                barrier.previousVersion
                    ? m_graphResourceRegistry.previousEntry(barrier.resource)
                    : m_graphResourceRegistry.entry(barrier.resource);
            if (!entry || !entry->buffer)
            {
                spdlog::error(
                    "[VulkanBackend] Missing transient graph buffer for barrier resource {}",
                    barrier.resource);
                return;
            }

            VkBufferMemoryBarrier bufferBarrier{};
            bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            bufferBarrier.srcAccessMask =
                (crossQueueAcquire || barrier.firstUse) ? 0 :
                accessMaskFor(barrier.oldUsage, barrier.fromAccess);
            bufferBarrier.dstAccessMask =
                releaseOnly ? 0 :
                accessMaskFor(barrier.newUsage, barrier.toAccess);
            bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bufferBarrier.buffer = entry->buffer.buffer;
            bufferBarrier.offset = 0;
            bufferBarrier.size = entry->buffer.size;

            VkPipelineStageFlags srcStage = (crossQueueAcquire || barrier.firstUse)
                ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                : pipelineStageFor(barrier.oldUsage, barrier.fromAccess);
            VkPipelineStageFlags dstStage =
                pipelineStageFor(barrier.newUsage, barrier.toAccess);
            if (commandQueue == QueueType::Compute)
            {
                srcStage = crossQueueAcquire ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                    : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
                dstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            }
            else if (commandQueue == QueueType::Transfer)
            {
                srcStage = crossQueueAcquire ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                    : VK_PIPELINE_STAGE_TRANSFER_BIT;
                dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            }

            vkCmdPipelineBarrier(
                cmd,
                srcStage,
                dstStage,
                0,
                0, nullptr,
                1, &bufferBarrier,
                0, nullptr);
            return;
        }

        VkImage image = VK_NULL_HANDLE;

        bool isSwapchainImage = false;

        if (resource.ownership == ResourceOwnership::Imported)
        {
            switch (resource.imported)
            {
            case ImportedResource::Swapchain:
                image = swapchainImage;
                isSwapchainImage = true;
                break;
            }
        }
        else
        {
            VulkanGraphResourceEntry* entry =
                barrier.previousVersion
                    ? m_graphResourceRegistry.previousEntry(barrier.resource)
                    : m_graphResourceRegistry.entry(barrier.resource);
            if (!entry || entry->type != GraphResourceType::Texture ||
                !entry->texture)
            {
                spdlog::error(
                    "[VulkanBackend] Missing transient graph texture for barrier resource {}",
                    barrier.resource);
                return;
            }
            image = entry->texture.image;
        }
        vkBarrier.image = image;

        VulkanGraphResourceEntry* trackedEntry =
            resource.ownership != ResourceOwnership::Imported
                ? (barrier.previousVersion
                    ? m_graphResourceRegistry.previousEntry(barrier.resource)
                    : m_graphResourceRegistry.entry(barrier.resource))
                : nullptr;
        const bool isExternalFirstUse =
            barrier.firstUse && !trackedEntry;

        vkBarrier.oldLayout = trackedEntry
            ? trackedEntry->layout
            : isExternalFirstUse
            ? (isSwapchainImage
                ? swapchainInitialLayout
                : VK_IMAGE_LAYOUT_UNDEFINED)
            : usageToLayout(barrier.oldUsage);

        vkBarrier.newLayout = usageToLayout(barrier.newUsage);

        vkBarrier.srcAccessMask =
            crossQueueAcquire || vkBarrier.oldLayout == VK_IMAGE_LAYOUT_UNDEFINED ||
                barrier.firstUse && !trackedEntry
                ? 0
                : accessMaskFor(
                    barrier.oldUsage,
                    barrier.fromAccess);

        vkBarrier.dstAccessMask =
            releaseOnly ? 0 : accessMaskFor(
                barrier.newUsage,
                barrier.toAccess);

        vkBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vkBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        // Native graph resources and swapchain images are concurrently shared;
        // queue-family ownership indices must remain IGNORED.

        const bool depthTexture =
            resource.type == GraphResourceType::Texture &&
            (resource.textureDesc.format == TextureFormat::D32_Float ||
                hasFlag(
                    resource.textureDesc.usage,
                    TextureUsageFlags::DepthAttachment));
        vkBarrier.subresourceRange.aspectMask =
            depthTexture ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;

        vkBarrier.subresourceRange.baseMipLevel = 0;
        vkBarrier.subresourceRange.levelCount =
            resource.ownership != ResourceOwnership::Imported
                ? std::max(1u, resource.textureDesc.mipLevels)
                : 1u;
        vkBarrier.subresourceRange.baseArrayLayer = 0;
        vkBarrier.subresourceRange.layerCount = 1;

        VkPipelineStageFlags srcStage =
            pipelineStageFor(
                barrier.oldUsage,
                barrier.fromAccess);

        if (vkBarrier.oldLayout == VK_IMAGE_LAYOUT_UNDEFINED)
        {
            srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        }
        if (crossQueueAcquire)
        {
            srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        }

        VkPipelineStageFlags dstStage =
            pipelineStageFor(
                barrier.newUsage,
                barrier.toAccess);

        if (barrier.newUsage == ResourceUsage::Present)
        {
            dstStage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        }
        if (commandQueue == QueueType::Compute)
        {
            srcStage = crossQueueAcquire ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            dstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        }
        else if (commandQueue == QueueType::Transfer)
        {
            srcStage = crossQueueAcquire ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                : VK_PIPELINE_STAGE_TRANSFER_BIT;
            dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }

        vkCmdPipelineBarrier(
            cmd,
            srcStage,
            dstStage,
            0,
            0, nullptr,
            0, nullptr,
            1, &vkBarrier);

        if (trackedEntry)
        {
            trackedEntry->layout = vkBarrier.newLayout;
        }

    }

    VulkanPassContext VulkanBackend::makePassContext(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node,
        const FrameContext& ctx,
        const SceneRenderView& scene,
        VkCommandBuffer cmd)
    {
        VulkanPassContext passCtx{};
        passCtx.cmd = cmd;
        passCtx.plan = &plan;
        passCtx.node = &node;
        passCtx.frame = &ctx;
        passCtx.scene = &scene;
        passCtx.resources = &m_graphResourceRegistry;
        passCtx.device = m_device.device();
        passCtx.allocator = &m_resourceAllocator;
        passCtx.pipelines = &m_pipelineManager;
        passCtx.gpuScene = &m_gpuScene;
        passCtx.surfaceExtent = m_swapchain.extent();
        return passCtx;
    }

    void VulkanBackend::dispatchNode(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node,
        const FrameContext& ctx,
        const SceneRenderView& scene,
        VkCommandBuffer cmd,
        VkImage swapchainImage)
    {
        if (node.payloadIndex >= plan.payloads.size())
        {
            return;
        }

        // Exhaustive payload dispatch: the visit binds the node's variant to the
        // matching recordPassPayload overload. Overload resolution replaces the
        // former node.type switch and the compute-node get_if cascade in one
        // place, and a new PassPayload alternative without an overload is a
        // compile error instead of a silently skipped pass.
        std::visit(
            [&](const auto& payload)
            {
                recordPassPayload(
                    payload, plan, node, ctx, scene, cmd, swapchainImage);
            },
            plan.payloads[node.payloadIndex]);
    }

    void VulkanBackend::recordPassPayload(
        const GraphicsPassData&,
        const CompiledGraphPlan& plan, const ExecutionNode& node,
        const FrameContext& ctx, const SceneRenderView& scene,
        VkCommandBuffer cmd, VkImage swapchainImage)
    {
        executeGraphicsNode(plan, node, ctx, scene, cmd, swapchainImage);
    }

    void VulkanBackend::recordPassPayload(
        const ComputePassData&,
        const CompiledGraphPlan& plan, const ExecutionNode& node,
        const FrameContext& ctx, const SceneRenderView& scene,
        VkCommandBuffer cmd, VkImage)
    {
        executeComputeNode(plan, node, ctx, scene, cmd);
    }

    void VulkanBackend::recordPassPayload(
        const PathTracePassData&,
        const CompiledGraphPlan& plan, const ExecutionNode& node,
        const FrameContext& ctx, const SceneRenderView& scene,
        VkCommandBuffer cmd, VkImage)
    {
        executePathTraceNode(plan, node, ctx, scene, cmd);
    }

    void VulkanBackend::recordPassPayload(
        const TonemapPassData&,
        const CompiledGraphPlan& plan, const ExecutionNode& node,
        const FrameContext& ctx, const SceneRenderView&,
        VkCommandBuffer cmd, VkImage)
    {
        executeTonemapNode(plan, node, ctx, cmd);
    }

    void VulkanBackend::recordPassPayload(
        const EnvironmentConvertPassData&,
        const CompiledGraphPlan& plan, const ExecutionNode& node,
        const FrameContext& ctx, const SceneRenderView& scene,
        VkCommandBuffer cmd, VkImage)
    {
        executeEnvironmentConvertNode(plan, node, ctx, scene, cmd);
    }

    void VulkanBackend::recordPassPayload(
        const TransferPassData&,
        const CompiledGraphPlan& plan, const ExecutionNode& node,
        const FrameContext& ctx, const SceneRenderView&,
        VkCommandBuffer cmd, VkImage swapchainImage)
    {
        executeTransferNode(plan, node, ctx, cmd, swapchainImage);
    }

    void VulkanBackend::executeGraphicsNode(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node,
        const FrameContext& ctx,
        const SceneRenderView& scene,
        VkCommandBuffer cmd,
        [[maybe_unused]] VkImage swapchainImage)
    {
        const GraphicsPipelineHandle pipelineHandle =
            pipelineForNode(plan, node);
        VulkanGraphicsPipeline* pipeline =
            m_pipelineManager.graphicsPipeline(pipelineHandle);
        if (!pipeline)
        {
            return;
        }

        const GraphicsPassData* pass =
            node.payloadIndex < plan.payloads.size()
                ? std::get_if<GraphicsPassData>(
                    &plan.payloads[node.payloadIndex])
                : nullptr;

        const bool hasColorTarget =
            pipeline->desc.colorAttachmentCount > 0;
        const GraphResourceId colorResource =
            findNodeResource(plan, node, ResourceUsage::ColorAttachment);
        const GraphResourceId depthResource =
            findNodeResource(plan, node, ResourceUsage::DepthAttachment);
        const VulkanGraphResourceEntry* colorEntry =
            colorResource != InvalidGraphResourceId
                ? m_graphResourceRegistry.entry(colorResource)
                : nullptr;
        const VulkanGraphResourceEntry* depthEntry =
            depthResource != InvalidGraphResourceId
                ? m_graphResourceRegistry.entry(depthResource)
                : nullptr;

        constexpr bool useGraphAttachments = true;
        ensureDepthTarget();

        if (pass && pass->drawList == DrawListKind::Skybox &&
            !m_environmentResources.converted)
        {
            if (VulkanComputePipeline* convertPipeline =
                    environmentConvertPipeline())
            {
                (void)convertEnvironmentIfReady(
                    *convertPipeline,
                    ctx,
                    scene,
                    cmd);
            }
        }

        VkClearValue clear{};
        clear.color = { { 0.02f, 0.02f, 0.025f, 1.0f } };

        if (!useGraphAttachments &&
            m_depthLayout != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        {
            VkImageMemoryBarrier depthBarrier{};
            depthBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            depthBarrier.oldLayout = m_depthLayout;
            depthBarrier.newLayout =
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depthBarrier.dstAccessMask =
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            depthBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            depthBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            depthBarrier.image = m_depthTexture.image;
            depthBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            depthBarrier.subresourceRange.levelCount = 1;
            depthBarrier.subresourceRange.layerCount = 1;
            vkCmdPipelineBarrier(
                cmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                0,
                0, nullptr,
                0, nullptr,
                1, &depthBarrier);
            m_depthLayout =
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        }

        VkRenderingAttachmentInfo colorAttachment{};
        colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttachment.imageView =
            useGraphAttachments &&
                colorEntry && colorEntry->ownership != ResourceOwnership::Imported
                ? colorEntry->view
                : m_swapchain.imageView(m_frameExecutor.currentSwapchainImage());
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp =
            pass && pass->colorLoadOp == AttachmentLoadOp::Load
                ? VK_ATTACHMENT_LOAD_OP_LOAD
                : (pass && pass->colorLoadOp == AttachmentLoadOp::DontCare
                    ? VK_ATTACHMENT_LOAD_OP_DONT_CARE
                    : VK_ATTACHMENT_LOAD_OP_CLEAR);
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.clearValue = clear;

        VkClearValue depthClear{};
        depthClear.depthStencil = { 1.0f, 0 };

        VkRenderingAttachmentInfo depthAttachment{};
        depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depthAttachment.imageView =
            useGraphAttachments &&
                depthEntry && depthEntry->ownership != ResourceOwnership::Imported
                ? depthEntry->view
                : m_depthImageView;
        depthAttachment.imageLayout =
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthAttachment.loadOp =
            pass && pass->depthLoadOp == AttachmentLoadOp::Load
                ? VK_ATTACHMENT_LOAD_OP_LOAD
                : (pass && pass->depthLoadOp == AttachmentLoadOp::DontCare
                    ? VK_ATTACHMENT_LOAD_OP_DONT_CARE
                    : VK_ATTACHMENT_LOAD_OP_CLEAR);
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depthAttachment.clearValue = depthClear;

        VkRenderingInfo renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderingInfo.renderArea = { {0, 0}, m_swapchain.extent() };
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = hasColorTarget ? 1u : 0u;
        renderingInfo.pColorAttachments =
            hasColorTarget ? &colorAttachment : nullptr;
        renderingInfo.pDepthAttachment = &depthAttachment;

        if (pass &&
            pass->drawList == DrawListKind::SceneGeometry &&
            !m_environmentResources.converted)
        {
            if (VulkanComputePipeline* convertPipeline =
                    environmentConvertPipeline())
            {
                static_cast<void>(convertEnvironmentIfReady(
                    *convertPipeline,
                    ctx,
                    scene,
                    cmd));
            }
        }

        vkCmdBeginRendering(cmd, &renderingInfo);

        if (pass &&
            pass->drawList == DrawListKind::SceneGeometry &&
            prepareSceneResources(ctx, scene, pipelineHandle) &&
            !m_gpuScene.draws().empty())
        {
            const uint32_t frameSlot =
                static_cast<uint32_t>(
                    ctx.frameIndex % m_gpuScene.frameSlotCount());
            VulkanGpuSceneFrameResources& frameResources =
                m_gpuScene.frameResources(frameSlot);

            VulkanForwardSceneInputs sceneInputs{};
            sceneInputs.pipeline = pipeline;
            sceneInputs.descriptorSet = frameResources.descriptorSet;
            sceneInputs.useGpuDriven =
                m_gpuDrivenBuffers.valid() &&
                !m_gpuScene.geometryBins().empty();
            sceneInputs.indirectStream.indirectArguments =
                m_gpuDrivenBuffers.indirectArguments;
            sceneInputs.indirectStream.binCounts = m_gpuDrivenBuffers.binCounts;
            sceneInputs.draws = m_gpuScene.draws();
            sceneInputs.geometryBins = m_gpuScene.geometryBins();
            sceneInputs.resolveNativeModel =
                [this](AssetHandle handle) -> VulkanUploadedModel*
                {
                    auto it = m_uploadedModels.find(handle);
                    return it != m_uploadedModels.end() ? &it->second : nullptr;
                };

            VulkanPassContext passCtx =
                makePassContext(plan, node, ctx, scene, cmd);
            recordForwardScene(passCtx, sceneInputs);
        }
        else if (pass && pass->drawList == DrawListKind::Skybox)
        {
            drawSkybox(*pipeline, ctx, scene, cmd);
        }

        vkCmdEndRendering(cmd);

    }

    void VulkanBackend::executeComputeNode(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node,
        [[maybe_unused]] const FrameContext& ctx,
        [[maybe_unused]] const SceneRenderView& scene,
        [[maybe_unused]] VkCommandBuffer cmd)
    {
        // dispatchNode's std::visit routes PathTrace/Tonemap/EnvironmentConvert
        // payloads to their own recorders, so this node is now only reached for a
        // generic ComputePassData; the per-layout decision below is owned here.
        const ComputePassData* pass =
            std::get_if<ComputePassData>(&plan.payloads[node.payloadIndex]);
        if (!pass || !pass->pipeline)
        {
            return;
        }

        VulkanComputePipeline* pipeline =
            m_pipelineManager.computePipeline(
                computePipelineForNode(plan, node));
        if (!pipeline)
        {
            return;
        }

        vkCmdBindPipeline(
            cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline->pipeline);

        if (pipeline->desc.bindingLayout ==
            PipelineBindingLayoutKind::ComputeStorageBuffer)
        {
            ensureComputeTestResources(*pipeline);

            VulkanPassContext passCtx{};
            passCtx.cmd = cmd;
            VulkanComputeStorageBufferTest test{};
            test.pipelineLayout = pipeline->pipelineLayout;
            test.descriptorSet = m_computeTestDescriptorSet;
            test.buffer = m_computeTestBuffer.buffer;
            test.bufferSize = m_computeTestBuffer.size;
            test.groupCountX = pass->groupCountX;
            test.groupCountY = pass->groupCountY;
            test.groupCountZ = pass->groupCountZ;
            recordComputeStorageBufferTest(passCtx, test);
            return;
        }
        else if (pipeline->desc.bindingLayout ==
            PipelineBindingLayoutKind::HiZDepthPyramid)
        {
            // Precondition: the forward pipeline must be resolvable and the
            // per-frame scene resources present (matches the original guard).
            const PipelineId graphicsPipelineId =
                makePipelineId("forward_bindless");
            const GraphicsPipelineHandle graphicsHandle = m_pipelineLibrary
                ? m_pipelineManager.resolveGraphicsPipeline(
                    *m_pipelineLibrary,
                    graphicsPipelineId,
                    swapchainTextureFormat())
                : GraphicsPipelineHandle{};
            if (!graphicsHandle || m_gpuScene.frameSlotCount() == 0)
            {
                return;
            }

            const uint32_t frameSlot = static_cast<uint32_t>(
                ctx.frameIndex % m_gpuScene.frameSlotCount());
            VulkanGpuSceneFrameResources& frameResources =
                m_gpuScene.frameResources(frameSlot);

            VulkanPassContext passCtx{};
            passCtx.cmd = cmd;
            passCtx.plan = &plan;
            passCtx.node = &node;
            passCtx.resources = &m_graphResourceRegistry;
            passCtx.device = m_device.device();

            VulkanHiZInputs hiZ{};
            hiZ.hiZId =
                findNodeResource(plan, node, ResourceUsage::StorageTexture);
            hiZ.sceneDepthId =
                findNodeResource(plan, node, ResourceUsage::SampledTexture);
            hiZ.hiZPool = &frameResources.hiZDescriptorPool;
            hiZ.frameConstants = frameResources.frameConstants
                ? frameResources.frameConstants.buffer
                : VK_NULL_HANDLE;
            hiZ.hiZDebugResourceOut =
                &m_clusteredForwardResources.hiZDebugResource;

            (void)recordHiZPyramid(passCtx, *pipeline, hiZ);
            return;
        }
        else if (pipeline->desc.bindingLayout ==
            PipelineBindingLayoutKind::GpuFrustumCull)
        {
            if (!prepareSceneResources(ctx, scene, {}, false) ||
                m_gpuScene.frameSlotCount() == 0)
            {
                return;
            }
            const uint32_t frameSlot = static_cast<uint32_t>(
                ctx.frameIndex % m_gpuScene.frameSlotCount());
            VulkanGpuSceneFrameResources& frameResources =
                m_gpuScene.frameResources(frameSlot);
            VulkanGpuScene& g = m_gpuScene;

            VulkanPassContext passCtx{};
            passCtx.cmd = cmd;
            passCtx.plan = &plan;
            passCtx.node = &node;
            passCtx.resources = &m_graphResourceRegistry;
            passCtx.device = m_device.device();

            // All cull buffers are graph-registry owned (resolved this frame in
            // resolveGpuDrivenBuffers); frameConstants is the backend per-frame CB.
            if (!m_gpuDrivenBuffers.valid())
            {
                return;
            }
            VulkanCullBuffers cull{};
            cull.gpuCullPool = &frameResources.gpuCullDescriptorPool;
            cull.frameConstants = frameResources.frameConstants.buffer;
            cull.instanceBounds = m_gpuDrivenBuffers.instanceBounds;
            cull.instanceBoundsSize = m_gpuDrivenBuffers.instanceBoundsSize;
            cull.visibleInstances = m_gpuDrivenBuffers.visibleInstances;
            cull.visibleInstancesSize = m_gpuDrivenBuffers.visibleInstancesSize;
            cull.visibleInstanceCount = m_gpuDrivenBuffers.visibleCount;
            cull.visibleInstanceCountSize = m_gpuDrivenBuffers.visibleCountSize;
            cull.drawInputs = m_gpuDrivenBuffers.drawInputs;
            cull.drawInputsSize = m_gpuDrivenBuffers.drawInputsSize;
            cull.indirectArguments = m_gpuDrivenBuffers.indirectArguments;
            cull.indirectArgumentsSize = m_gpuDrivenBuffers.indirectArgumentsSize;
            cull.drawMetadata = m_gpuDrivenBuffers.drawMetadata;
            cull.drawMetadataSize = m_gpuDrivenBuffers.drawMetadataSize;
            cull.binCounts = m_gpuDrivenBuffers.binCounts;
            cull.binCountsSize = m_gpuDrivenBuffers.binCountsSize;

            if (!recordGpuFrustumCull(passCtx, *pipeline, cull))
            {
                return;
            }

            if (!g.loggedGpuCull)
            {
                spdlog::info(
                    "[VulkanBackend] GPU frustum culling dispatch prepared for {} instance(s)",
                    m_gpuScene.instanceCount());
                g.loggedGpuCull = true;
            }
        }
        else if (pipeline->desc.bindingLayout ==
            PipelineBindingLayoutKind::ClusteredForward)
        {
            if (!bindClusteredForwardCompute(*pipeline, ctx, scene, cmd))
            {
                return;
            }
        }

        if (pipeline->desc.bindingLayout ==
            PipelineBindingLayoutKind::GpuFrustumCull)
        {
            const uint32_t instanceCount =
                std::min<uint32_t>(
                    m_gpuScene.instanceCount(),
                    ClusteredForwardMaxGpuCullInstances);
            vkCmdDispatch(
                cmd,
                std::max(1u, (instanceCount + 63u) / 64u),
                1,
                1);
        }
        else
        {
            vkCmdDispatch(
                cmd,
                pass->groupCountX,
                pass->groupCountY,
                pass->groupCountZ);
        }

        // The clustered cluster buffers and the GPU-driven cull buffers are all
        // frame-graph owned: the graph emits the write->read buffer barriers
        // (including the cross-queue cluster-compute -> graphics handoff and the
        // DRAW_INDIRECT / INDIRECT_COMMAND_READ scope) from the declared
        // dependencies, so no manual post-dispatch barrier is recorded here.
    }

    void VulkanBackend::executeEnvironmentConvertNode(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node,
        const FrameContext& ctx,
        const SceneRenderView& scene,
        VkCommandBuffer cmd)
    {
        if (!ensureEnvironmentResources(ctx, scene, cmd) ||
            m_environmentResources.converted)
        {
            return;
        }

        VulkanComputePipeline* pipeline =
            m_pipelineManager.computePipeline(
                computePipelineForNode(plan, node));
        if (!pipeline)
        {
            return;
        }

        (void)convertEnvironmentIfReady(*pipeline, ctx, scene, cmd);
    }

    VulkanComputePipeline* VulkanBackend::environmentConvertPipeline()
    {
        if (!m_pipelineLibrary)
        {
            return nullptr;
        }

        const PipelineId pipelineId = makePipelineId("equirect_to_cubemap");
        const ComputePipelineHandle handle =
            m_pipelineManager.resolveComputePipeline(
                *m_pipelineLibrary, pipelineId);
        return m_pipelineManager.computePipeline(handle);
    }

    bool VulkanBackend::convertEnvironmentIfReady(
        VulkanComputePipeline& pipeline,
        const FrameContext& ctx,
        const SceneRenderView& scene,
        VkCommandBuffer cmd)
    {
        if (!ensureEnvironmentResources(ctx, scene, cmd) ||
            m_environmentResources.converted)
        {
            return m_environmentResources.converted;
        }

        const uint64_t key =
            uploadedTextureKey(
                scene.environment.equirectTexture,
                0,
                TextureTransferFunction::Linear);
        auto textureIt = m_uploadedTextures.find(key);
        if (textureIt == m_uploadedTextures.end())
        {
            return false;
        }

        if (m_environmentResources.convertDescriptorSet == VK_NULL_HANDLE)
        {
            if (m_environmentResources.bakeDescriptorPool == VK_NULL_HANDLE)
            {
                VkDescriptorPoolSize bakePoolSizes[3]{};
                bakePoolSizes[0].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                bakePoolSizes[0].descriptorCount = 64;
                bakePoolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                bakePoolSizes[1].descriptorCount = 64;
                bakePoolSizes[2].type = VK_DESCRIPTOR_TYPE_SAMPLER;
                bakePoolSizes[2].descriptorCount = 64;

                VkDescriptorPoolCreateInfo bakePoolInfo{};
                bakePoolInfo.sType =
                    VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                bakePoolInfo.maxSets = 64;
                bakePoolInfo.poolSizeCount =
                    static_cast<uint32_t>(std::size(bakePoolSizes));
                bakePoolInfo.pPoolSizes = bakePoolSizes;
                throwIfFailed(
                    vkCreateDescriptorPool(
                        m_device.device(),
                        &bakePoolInfo,
                        nullptr,
                        &m_environmentResources.bakeDescriptorPool),
                    "Failed to create Vulkan IBL bake descriptor pool.");
            }

            VkDescriptorSetAllocateInfo allocateInfo{};
            allocateInfo.sType =
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocateInfo.descriptorPool =
                m_environmentResources.bakeDescriptorPool;
            allocateInfo.descriptorSetCount = 1;
            allocateInfo.pSetLayouts = &pipeline.descriptorSetLayout;
            throwIfFailed(
                vkAllocateDescriptorSets(
                    m_device.device(),
                    &allocateInfo,
                    &m_environmentResources.convertDescriptorSet),
                "Failed to allocate Vulkan environment conversion descriptor set.");

            VkDescriptorImageInfo sourceInfo{};
            sourceInfo.imageView = textureIt->second.view;
            sourceInfo.imageLayout =
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorImageInfo outputInfo{};
            outputInfo.imageView =
                m_environmentResources.cubemapStorageView;
            outputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkDescriptorImageInfo samplerInfo{};
            samplerInfo.sampler = m_environmentResources.sampler;

            VkWriteDescriptorSet writes[3]{};
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet =
                m_environmentResources.convertDescriptorSet;
            writes[0].dstBinding = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            writes[0].pImageInfo = &sourceInfo;

            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet =
                m_environmentResources.convertDescriptorSet;
            writes[1].dstBinding = 1;
            writes[1].descriptorCount = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[1].pImageInfo = &outputInfo;

            writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[2].dstSet =
                m_environmentResources.convertDescriptorSet;
            writes[2].dstBinding = 2;
            writes[2].descriptorCount = 1;
            writes[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            writes[2].pImageInfo = &samplerInfo;

            vkUpdateDescriptorSets(
                m_device.device(),
                static_cast<uint32_t>(std::size(writes)),
                writes,
                0,
                nullptr);
        }

        if (m_environmentResources.cubemapLayout !=
            VK_IMAGE_LAYOUT_GENERAL)
        {
            VkImageMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = m_environmentResources.cubemapLayout;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = m_environmentResources.cubemap.image;
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.layerCount = 6;

            vkCmdPipelineBarrier(
                cmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr,
                0, nullptr,
                1,
                &barrier);
            m_environmentResources.cubemapLayout = VK_IMAGE_LAYOUT_GENERAL;
        }

        VulkanPassContext convertCtx{};
        convertCtx.cmd = cmd;
        recordEnvironmentConvert(
            convertCtx,
            pipeline,
            m_environmentResources.convertDescriptorSet,
            m_environmentResources.cubemapSize);

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = m_environmentResources.cubemap.image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 6;

        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1,
            &barrier);

        m_environmentResources.cubemapLayout =
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        m_environmentResources.converted = true;
        m_environmentResources.skyboxDescriptorsDirty = true;
        ++m_pathTraceResources.pathTraceDescriptorGeneration;
        m_pathTraceResources.pathTraceDescriptorsDirty = true;
        spdlog::info("[Vulkan] Converted HDR environment to cubemap");
        return true;
    }

    bool VulkanBackend::ensureEnvironmentResources(
        const FrameContext& ctx,
        const SceneRenderView& scene,
        VkCommandBuffer cmd)
    {
        (void)cmd;
        if (scene.environment.enabled == 0u ||
            !scene.environment.equirectTexture ||
            !ctx.services ||
            !ctx.services->assetManager)
        {
            return false;
        }

        const uint32_t cubemapSize =
            std::max(1u, scene.environment.settings.cubemapSize);
        if (!m_environmentResources.source)
        {
            // IBL baking may populate the shared environment textures before
            // the first graph conversion pass records. Adopt that source
            // instead of destroying fresh resources and waiting the device.
            m_environmentResources.source = scene.environment.equirectTexture;
            m_environmentResources.cubemapSize = cubemapSize;
        }
        if (m_environmentResources.source != scene.environment.equirectTexture ||
            m_environmentResources.cubemapSize != cubemapSize)
        {
            destroyEnvironmentResources();
            m_environmentResources.source = scene.environment.equirectTexture;
            m_environmentResources.cubemapSize = cubemapSize;
            m_pathTraceResources.accumulatedSampleCount = 0;
            m_pathTraceResources.resetAccumulation = true;
        }

        if (!m_environmentResources.cubemap)
        {
            m_environmentResources.cubemap =
                m_resourceAllocator.createTexture({
                    .width = m_environmentResources.cubemapSize,
                    .height = m_environmentResources.cubemapSize,
                    .depth = 1,
                    .mipLevels = 1,
                    .arrayLayers = 6,
                    .cubeCompatible = true,
                    .format = TextureFormat::RGBA32_Float,
                    .usage =
                        TextureUsageFlags::Sampled |
                        TextureUsageFlags::Storage,
                    .memoryUsage = ResourceMemoryUsage::GpuOnly,
                    .debugName = "Vulkan environment cubemap"
                });
            m_environmentResources.cubemapLayout =
                VK_IMAGE_LAYOUT_UNDEFINED;

            VkImageViewCreateInfo cubeView{};
            cubeView.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            cubeView.image = m_environmentResources.cubemap.image;
            cubeView.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
            cubeView.format = m_environmentResources.cubemap.format;
            cubeView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            cubeView.subresourceRange.levelCount = 1;
            cubeView.subresourceRange.layerCount = 6;
            throwIfFailed(
                vkCreateImageView(
                    m_device.device(),
                    &cubeView,
                    nullptr,
                    &m_environmentResources.cubemapView),
                "Failed to create Vulkan environment cubemap view.");

            VkImageViewCreateInfo storageView = cubeView;
            storageView.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
            throwIfFailed(
                vkCreateImageView(
                    m_device.device(),
                    &storageView,
                    nullptr,
                    &m_environmentResources.cubemapStorageView),
                "Failed to create Vulkan environment storage view.");

            VkSamplerCreateInfo sampler{};
            sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sampler.magFilter = VK_FILTER_LINEAR;
            sampler.minFilter = VK_FILTER_LINEAR;
            sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sampler.maxLod = VK_LOD_CLAMP_NONE;
            throwIfFailed(
                vkCreateSampler(
                    m_device.device(),
                    &sampler,
                    nullptr,
                    &m_environmentResources.sampler),
                "Failed to create Vulkan environment sampler.");

            VkImageMemoryBarrier fallbackReadBarrier{};
            fallbackReadBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            fallbackReadBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            fallbackReadBarrier.newLayout =
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            fallbackReadBarrier.srcAccessMask = 0;
            fallbackReadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            fallbackReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            fallbackReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            fallbackReadBarrier.image = m_environmentResources.cubemap.image;
            fallbackReadBarrier.subresourceRange.aspectMask =
                VK_IMAGE_ASPECT_COLOR_BIT;
            fallbackReadBarrier.subresourceRange.levelCount = 1;
            fallbackReadBarrier.subresourceRange.layerCount = 6;
            vkCmdPipelineBarrier(
                cmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0,
                nullptr,
                0,
                nullptr,
                1,
                &fallbackReadBarrier);
            m_environmentResources.cubemapLayout =
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        if (m_environmentResources.skyboxConstants.empty())
        {
            const uint32_t frameCount =
                static_cast<uint32_t>(
                    std::max<size_t>(1, m_frameExecutor.framesInFlight()));
            m_environmentResources.skyboxConstants.resize(frameCount);
            m_environmentResources.skyboxDescriptorSets.assign(
                frameCount,
                VK_NULL_HANDLE);
            for (VulkanBuffer& buffer :
                m_environmentResources.skyboxConstants)
            {
                buffer = m_resourceAllocator.createBuffer({
                    .size = sizeof(SkyboxConstants),
                    .usage = BufferUsageFlags::Constant,
                    .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                    .mappedAtCreation = true,
                    .debugName = "Vulkan skybox constants"
                });
            }
        }

        if (m_environmentResources.descriptorPool == VK_NULL_HANDLE)
        {
            const uint32_t frameCount =
                static_cast<uint32_t>(
                    std::max<size_t>(
                        1,
                        m_environmentResources.skyboxConstants.size()));
            VkDescriptorPoolSize poolSizes[4]{};
            poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            poolSizes[0].descriptorCount = frameCount;
            poolSizes[1].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            poolSizes[1].descriptorCount = frameCount + 64u;
            poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            poolSizes[2].descriptorCount = 64u;
            poolSizes[3].type = VK_DESCRIPTOR_TYPE_SAMPLER;
            poolSizes[3].descriptorCount = frameCount + 64u;

            VkDescriptorPoolCreateInfo poolInfo{};
            poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.maxSets = frameCount + 64u;
            poolInfo.poolSizeCount =
                static_cast<uint32_t>(std::size(poolSizes));
            poolInfo.pPoolSizes = poolSizes;
            throwIfFailed(
                vkCreateDescriptorPool(
                    m_device.device(),
                    &poolInfo,
                    nullptr,
                    &m_environmentResources.descriptorPool),
                "Failed to create Vulkan environment descriptor pool.");
        }

        if (m_environmentResources.converted)
        {
            return true;
        }

        const ImageAsset* image =
            ctx.services->assetManager->image(scene.environment.equirectTexture);
        if (!image || !image->valid())
        {
            return true;
        }

        requestTexture(
            scene.environment.equirectTexture,
            0,
            *image,
            TextureTransferFunction::Linear);

        return true;
    }

    void VulkanBackend::updateSkyboxDescriptors(
        const VulkanGraphicsPipeline& pipeline)
    {
        if (!m_environmentResources.converted ||
            m_environmentResources.descriptorPool == VK_NULL_HANDLE ||
            m_environmentResources.skyboxConstants.empty())
        {
            return;
        }

        const bool needAllocate =
            m_environmentResources.skyboxDescriptorSets.empty() ||
            std::ranges::any_of(
                m_environmentResources.skyboxDescriptorSets,
                [](VkDescriptorSet set)
                {
                    return set == VK_NULL_HANDLE;
                });

        if (needAllocate)
        {
            const uint32_t setCount =
                static_cast<uint32_t>(
                    m_environmentResources.skyboxDescriptorSets.size());
            std::vector<VkDescriptorSetLayout> layouts(
                setCount,
                pipeline.descriptorSetLayout);

            VkDescriptorSetAllocateInfo allocateInfo{};
            allocateInfo.sType =
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocateInfo.descriptorPool =
                m_environmentResources.descriptorPool;
            allocateInfo.descriptorSetCount = setCount;
            allocateInfo.pSetLayouts = layouts.data();
            throwIfFailed(
                vkAllocateDescriptorSets(
                    m_device.device(),
                    &allocateInfo,
                    m_environmentResources.skyboxDescriptorSets.data()),
                "Failed to allocate Vulkan skybox descriptor sets.");
            m_environmentResources.skyboxDescriptorsDirty = true;
        }

        if (!m_environmentResources.skyboxDescriptorsDirty)
        {
            return;
        }

        VkDescriptorImageInfo cubemapInfo{};
        cubemapInfo.imageView = m_environmentResources.cubemapView;
        cubemapInfo.imageLayout =
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo samplerInfo{};
        samplerInfo.sampler = m_environmentResources.sampler;

        std::vector<VkDescriptorBufferInfo> constantsInfos(
            m_environmentResources.skyboxConstants.size());
        std::vector<VkWriteDescriptorSet> writes;
        writes.reserve(
            m_environmentResources.skyboxDescriptorSets.size() * 3u);

        for (size_t setIndex = 0;
             setIndex < m_environmentResources.skyboxDescriptorSets.size();
             ++setIndex)
        {
            constantsInfos[setIndex].buffer =
                m_environmentResources.skyboxConstants[setIndex].buffer;
            constantsInfos[setIndex].range = sizeof(SkyboxConstants);

            VkWriteDescriptorSet constantsWrite{};
            constantsWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            constantsWrite.dstSet =
                m_environmentResources.skyboxDescriptorSets[setIndex];
            constantsWrite.dstBinding = 0;
            constantsWrite.descriptorCount = 1;
            constantsWrite.descriptorType =
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            constantsWrite.pBufferInfo = &constantsInfos[setIndex];
            writes.push_back(constantsWrite);

            VkWriteDescriptorSet cubemapWrite{};
            cubemapWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            cubemapWrite.dstSet =
                m_environmentResources.skyboxDescriptorSets[setIndex];
            cubemapWrite.dstBinding = 1;
            cubemapWrite.descriptorCount = 1;
            cubemapWrite.descriptorType =
                VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            cubemapWrite.pImageInfo = &cubemapInfo;
            writes.push_back(cubemapWrite);

            VkWriteDescriptorSet samplerWrite{};
            samplerWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            samplerWrite.dstSet =
                m_environmentResources.skyboxDescriptorSets[setIndex];
            samplerWrite.dstBinding = 100;
            samplerWrite.descriptorCount = 1;
            samplerWrite.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            samplerWrite.pImageInfo = &samplerInfo;
            writes.push_back(samplerWrite);
        }

        vkUpdateDescriptorSets(
            m_device.device(),
            static_cast<uint32_t>(writes.size()),
            writes.data(),
            0,
            nullptr);

        m_environmentResources.skyboxDescriptorsDirty = false;
    }

    void VulkanBackend::drawSkybox(
        VulkanGraphicsPipeline& pipeline,
        const FrameContext& ctx,
        const SceneRenderView& scene,
        VkCommandBuffer cmd)
    {
        if (!ensureEnvironmentResources(ctx, scene, cmd) ||
            !m_environmentResources.converted ||
            m_environmentResources.skyboxConstants.empty())
        {
            return;
        }

        updateSkyboxDescriptors(pipeline);

        const uint32_t frameSlot =
            static_cast<uint32_t>(
                ctx.frameIndex %
                m_environmentResources.skyboxConstants.size());
        if (frameSlot >= m_environmentResources.skyboxDescriptorSets.size())
        {
            return;
        }

        const VkDescriptorSet descriptorSet =
            m_environmentResources.skyboxDescriptorSets[frameSlot];
        if (descriptorSet == VK_NULL_HANDLE)
        {
            return;
        }

        VulkanSkyboxInputs inputs{};
        inputs.pipeline = &pipeline;
        inputs.descriptorSet = descriptorSet;
        inputs.constants = &m_environmentResources.skyboxConstants[frameSlot];

        VulkanPassContext passCtx{};
        passCtx.cmd = cmd;
        passCtx.scene = &scene;
        passCtx.allocator = &m_resourceAllocator;
        passCtx.surfaceExtent = m_swapchain.extent();
        recordSkybox(passCtx, inputs);
    }

    void VulkanBackend::destroyEnvironmentResources()
    {
        m_retirementQueue.retireImageView(m_environmentResources.cubemapView);
        m_retirementQueue.retireImageView(
            m_environmentResources.cubemapStorageView);
        m_retirementQueue.retireImageView(
            m_environmentResources.irradianceView);
        m_retirementQueue.retireImageView(
            m_environmentResources.irradianceStorageView);
        m_retirementQueue.retireImageView(
            m_environmentResources.prefilteredView);
        for (VkImageView view : m_environmentResources.prefilteredStorageViews)
        {
            m_retirementQueue.retireImageView(view);
        }
        m_retirementQueue.retireImageView(m_environmentResources.brdfLutView);
        m_retirementQueue.retireImageView(
            m_environmentResources.brdfLutStorageView);
        m_retirementQueue.retireImageView(m_environmentResources.equirectView);
        m_retirementQueue.retireSampler(m_environmentResources.sampler);
        m_retirementQueue.retireDescriptorPool(
            m_environmentResources.descriptorPool);
        m_retirementQueue.retireDescriptorPool(
            m_environmentResources.bakeDescriptorPool);

        for (VulkanBuffer& buffer :
            m_environmentResources.skyboxConstants)
        {
            m_retirementQueue.retire(std::move(buffer));
        }
        m_retirementQueue.retire(std::move(m_environmentResources.cubemap));
        m_retirementQueue.retire(std::move(m_environmentResources.equirect));
        m_retirementQueue.retire(std::move(m_environmentResources.irradiance));
        m_retirementQueue.retire(std::move(m_environmentResources.prefiltered));
        m_retirementQueue.retire(std::move(m_environmentResources.brdfLut));
        m_environmentResources = {};
    }

    void VulkanBackend::executePathTraceNode(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node,
        const FrameContext& ctx,
        const SceneRenderView& scene,
        VkCommandBuffer cmd)
    {
        const ComputePipelineHandle pathTracePipelineHandle =
            computePipelineForNode(plan, node);

        ensurePathTraceResources();
        ensurePathTraceSceneResources(ctx, scene);
        const bool environmentResourcesAvailable =
            ensureEnvironmentResources(ctx, scene, cmd);
        if (!environmentResourcesAvailable)
        {
            return;
        }
        if (!m_environmentResources.converted)
        {
            if (VulkanComputePipeline* convertPipeline =
                    environmentConvertPipeline())
            {
                static_cast<void>(convertEnvironmentIfReady(
                    *convertPipeline,
                    ctx,
                    scene,
                    cmd));
            }
        }
        const bool environmentReady = m_environmentResources.converted;

        VulkanComputePipeline* pipeline =
            m_pipelineManager.computePipeline(pathTracePipelineHandle);
        if (!pipeline)
        {
            return;
        }
        updatePathTraceDescriptors(
            pipeline, nullptr,
            static_cast<uint32_t>(ctx.frameIndex %
                m_pathTraceResources.pathTraceDescriptorSets.size()));

        if (!m_pathTraceResources.accumulation ||
            m_pathTraceResources.pathTraceDescriptorSets.empty())
        {
            return;
        }

        const uint32_t frameSlot =
            static_cast<uint32_t>(
                ctx.frameIndex %
                m_pathTraceResources.pathTraceConstants.size());
        const VkDescriptorSet descriptorSet =
            m_pathTraceResources.pathTraceDescriptorSets[frameSlot];
        if (descriptorSet == VK_NULL_HANDLE)
        {
            return;
        }

        const bool hasCamera = scene.camera.valid != 0u;
        const bool cameraChanged =
            hasCamera &&
            (!m_pathTraceResources.hasPreviousCamera ||
                matricesDiffer(scene.camera.view, m_pathTraceResources.previousView) ||
                matricesDiffer(
                    scene.camera.projection,
                    m_pathTraceResources.previousProjection));

        if (cameraChanged)
        {
            m_pathTraceResources.accumulatedSampleCount = 0;
            m_pathTraceResources.resetAccumulation = true;
            m_pathTraceResources.previousView = scene.camera.view;
            m_pathTraceResources.previousProjection = scene.camera.projection;
            m_pathTraceResources.hasPreviousCamera = true;
        }
        else if (!hasCamera)
        {
            m_pathTraceResources.hasPreviousCamera = false;
        }

        if (m_pathTraceResources.environmentVersion !=
            scene.environment.version)
        {
            m_pathTraceResources.environmentVersion =
                scene.environment.version;
            m_pathTraceResources.accumulatedSampleCount = 0;
            m_pathTraceResources.resetAccumulation = true;
        }
        m_pathTraceResources.tonemapExposure =
            scene.environment.settings.tonemapExposure;

        if (m_pathTraceResources.accumulationLayout !=
            VK_IMAGE_LAYOUT_GENERAL)
        {
            transitionImage(
                cmd,
                m_pathTraceResources.accumulation.image,
                m_pathTraceResources.accumulationLayout,
                VK_IMAGE_LAYOUT_GENERAL,
                m_pathTraceResources.accumulationLayout ==
                    VK_IMAGE_LAYOUT_UNDEFINED
                        ? 0
                        : VK_ACCESS_SHADER_READ_BIT,
                VK_ACCESS_SHADER_WRITE_BIT,
                m_pathTraceResources.accumulationLayout ==
                    VK_IMAGE_LAYOUT_UNDEFINED
                        ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                        : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            m_pathTraceResources.accumulationLayout =
                VK_IMAGE_LAYOUT_GENERAL;
        }

        VulkanPathTraceInputs inputs{};
        inputs.pipeline = pipeline;
        inputs.descriptorSet = descriptorSet;
        inputs.constants = &m_pathTraceResources.pathTraceConstants[frameSlot];
        inputs.width = m_pathTraceResources.width;
        inputs.height = m_pathTraceResources.height;
        inputs.frameIndex = static_cast<uint32_t>(ctx.frameIndex);
        inputs.accumulatedSampleCount =
            m_pathTraceResources.accumulatedSampleCount;
        inputs.exposure = m_pathTraceResources.tonemapExposure;
        inputs.resetAccumulation = m_pathTraceResources.resetAccumulation;
        inputs.sceneVertexCount = m_pathTraceResources.sceneVertexCount;
        inputs.sceneMaterialCount = m_pathTraceResources.sceneMaterialCount;
        inputs.sceneTriangleCount = m_pathTraceResources.sceneTriangleCount;
        inputs.sceneBvhNodeCount = m_pathTraceResources.sceneBvhNodeCount;
        inputs.firstEmissiveTriangleIndex =
            m_pathTraceResources.firstEmissiveTriangleIndex;
        inputs.environmentReady = environmentReady;
        inputs.environmentIntensity = scene.environment.settings.intensity;
        inputs.environmentExposure =
            scene.environment.settings.pathTraceExposure;

        VulkanPassContext passCtx =
            makePassContext(plan, node, ctx, scene, cmd);
        recordPathTrace(passCtx, inputs);

        ++m_pathTraceResources.accumulatedSampleCount;
        m_pathTraceResources.resetAccumulation = false;
    }

    void VulkanBackend::executeTonemapNode(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node,
        const FrameContext& ctx,
        VkCommandBuffer cmd)
    {
        VulkanComputePipeline* pipeline =
            m_pipelineManager.computePipeline(
                computePipelineForNode(plan, node));
        if (!pipeline)
        {
            return;
        }

        ensurePathTraceResources();
        updatePathTraceDescriptors(
            nullptr, pipeline,
            static_cast<uint32_t>(ctx.frameIndex %
                m_pathTraceResources.tonemapDescriptorSets.size()));

        if (!m_pathTraceResources.accumulation ||
            !m_pathTraceResources.tonemap ||
            m_pathTraceResources.tonemapDescriptorSets.empty())
        {
            return;
        }

        const uint32_t frameSlot =
            static_cast<uint32_t>(
                ctx.frameIndex %
                m_pathTraceResources.tonemapConstants.size());
        const VkDescriptorSet descriptorSet =
            m_pathTraceResources.tonemapDescriptorSets[frameSlot];
        if (descriptorSet == VK_NULL_HANDLE)
        {
            return;
        }

        if (m_pathTraceResources.accumulationLayout !=
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        {
            transitionImage(
                cmd,
                m_pathTraceResources.accumulation.image,
                m_pathTraceResources.accumulationLayout,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                m_pathTraceResources.accumulationLayout ==
                    VK_IMAGE_LAYOUT_UNDEFINED
                        ? 0
                        : VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                m_pathTraceResources.accumulationLayout ==
                    VK_IMAGE_LAYOUT_UNDEFINED
                        ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                        : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            m_pathTraceResources.accumulationLayout =
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        if (m_pathTraceResources.tonemapLayout !=
            VK_IMAGE_LAYOUT_GENERAL)
        {
            transitionImage(
                cmd,
                m_pathTraceResources.tonemap.image,
                m_pathTraceResources.tonemapLayout,
                VK_IMAGE_LAYOUT_GENERAL,
                m_pathTraceResources.tonemapLayout ==
                    VK_IMAGE_LAYOUT_UNDEFINED
                        ? 0
                        : VK_ACCESS_TRANSFER_READ_BIT,
                VK_ACCESS_SHADER_WRITE_BIT,
                m_pathTraceResources.tonemapLayout ==
                    VK_IMAGE_LAYOUT_UNDEFINED
                        ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                        : VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            m_pathTraceResources.tonemapLayout =
                VK_IMAGE_LAYOUT_GENERAL;
        }

        VulkanTonemapInputs inputs{};
        inputs.pipeline = pipeline;
        inputs.descriptorSet = descriptorSet;
        inputs.constants = &m_pathTraceResources.tonemapConstants[frameSlot];
        inputs.width = m_pathTraceResources.width;
        inputs.height = m_pathTraceResources.height;
        inputs.exposure = m_pathTraceResources.tonemapExposure;

        VulkanPassContext passCtx{};
        passCtx.cmd = cmd;
        passCtx.allocator = &m_resourceAllocator;
        recordTonemap(passCtx, inputs);
    }

    void VulkanBackend::executeTransferNode(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node,
        [[maybe_unused]] const FrameContext& ctx,
        VkCommandBuffer cmd,
        VkImage swapchainImage)
    {
        if (node.payloadIndex >= plan.payloads.size())
        {
            return;
        }

        const TransferPassData* pass =
            std::get_if<TransferPassData>(&plan.payloads[node.payloadIndex]);
        if (!pass)
        {
            return;
        }

        if (pass->source >= plan.resources.size() ||
            pass->destination >= plan.resources.size())
        {
            spdlog::error("Transfer pass '{}' has invalid graph resources", pass->name);
            return;
        }

        const GraphResource& sourceDesc = plan.resources[pass->source];
        const GraphResource& destinationDesc = plan.resources[pass->destination];
        if (sourceDesc.type != destinationDesc.type)
        {
            spdlog::error("Transfer pass '{}' cannot copy between resource types", pass->name);
            return;
        }

        const VulkanGraphResourceEntry* sourceEntry =
            m_graphResourceRegistry.entry(sourceDesc.id);
        const VulkanGraphResourceEntry* destinationEntry =
            m_graphResourceRegistry.entry(destinationDesc.id);

        if (sourceDesc.type == GraphResourceType::Buffer)
        {
            if (!sourceEntry || !destinationEntry ||
                !sourceEntry->buffer || !destinationEntry->buffer)
            {
                spdlog::error("Transfer pass '{}' could not resolve its buffers", pass->name);
                return;
            }
            VulkanTransferCopy copy{};
            copy.isBuffer = true;
            copy.sourceBuffer = sourceEntry->buffer.buffer;
            copy.destinationBuffer = destinationEntry->buffer.buffer;
            copy.bufferSize = std::min(
                sourceEntry->buffer.size, destinationEntry->buffer.size);
            recordTransferCopy(cmd, copy);
            return;
        }

        VkImage sourceImage = sourceEntry ? sourceEntry->texture.image : VK_NULL_HANDLE;
        VkImage destinationImage = destinationEntry
            ? destinationEntry->texture.image : VK_NULL_HANDLE;
        uint32_t width = sourceEntry ? sourceEntry->width : 0;
        uint32_t height = sourceEntry ? sourceEntry->height : 0;
        if (sourceDesc.semantic == GraphResourceSemantic::PathTraceTonemap)
        {
            sourceImage = m_pathTraceResources.tonemap.image;
            width = m_pathTraceResources.width;
            height = m_pathTraceResources.height;
            if (m_pathTraceResources.tonemapLayout !=
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
            {
                transitionImage(
                    cmd,
                    m_pathTraceResources.tonemap.image,
                    m_pathTraceResources.tonemapLayout,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_ACCESS_SHADER_WRITE_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT);
                m_pathTraceResources.tonemapLayout =
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            }
        }
        if (destinationDesc.imported == ImportedResource::Swapchain)
        {
            destinationImage = swapchainImage;
        }
        if (sourceImage == VK_NULL_HANDLE || destinationImage == VK_NULL_HANDLE)
        {
            spdlog::error("Transfer pass '{}' could not resolve its images", pass->name);
            return;
        }
        if (destinationEntry &&
            destinationDesc.ownership != ResourceOwnership::Imported)
        {
            width = std::min(width, destinationEntry->width);
            height = std::min(height, destinationEntry->height);
        }
        if (width == 0 || height == 0)
        {
            spdlog::error("Transfer pass '{}' resolved a zero-sized image copy", pass->name);
            return;
        }

        VulkanTransferCopy copy{};
        copy.isBuffer = false;
        copy.sourceImage = sourceImage;
        copy.destinationImage = destinationImage;
        copy.width = width;
        copy.height = height;
        recordTransferCopy(cmd, copy);
    }

    GraphicsPipelineHandle VulkanBackend::pipelineForNode(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node)
    {
        if (!m_pipelineLibrary ||
            node.payloadIndex >= plan.payloads.size())
        {
            return {};
        }

        const GraphicsPassData* pass =
            std::get_if<GraphicsPassData>(&plan.payloads[node.payloadIndex]);
        if (!pass || !pass->pipeline)
        {
            return {};
        }

        return m_pipelineManager.resolveGraphicsPipeline(
            *m_pipelineLibrary, pass->pipeline, swapchainTextureFormat());
    }

    ComputePipelineHandle VulkanBackend::computePipelineForNode(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node)
    {
        if (!m_pipelineLibrary ||
            node.payloadIndex >= plan.payloads.size())
        {
            return {};
        }

        PipelineId pipelineId{};

        if (const ComputePassData* computePass =
            std::get_if<ComputePassData>(&plan.payloads[node.payloadIndex]))
        {
            pipelineId = computePass->pipeline;
        }
        else if (const PathTracePassData* pathTracePass =
            std::get_if<PathTracePassData>(&plan.payloads[node.payloadIndex]))
        {
            pipelineId = pathTracePass->pipeline;
        }
        else if (const TonemapPassData* tonemapPass =
            std::get_if<TonemapPassData>(&plan.payloads[node.payloadIndex]))
        {
            pipelineId = tonemapPass->pipeline;
        }

        if (!pipelineId)
        {
            return {};
        }

        return m_pipelineManager.resolveComputePipeline(
            *m_pipelineLibrary, pipelineId);
    }

    void VulkanBackend::destroySceneResources()
    {
        destroyDepthTarget();
        destroyPathTraceResources();
        destroyClusteredForwardResources();

        if (m_computeTestDescriptorPool != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(
                m_device.device(),
                m_computeTestDescriptorPool,
                nullptr);
            m_computeTestDescriptorPool = VK_NULL_HANDLE;
            m_computeTestDescriptorSet = VK_NULL_HANDLE;
        }
        m_resourceAllocator.destroyBuffer(m_computeTestBuffer);

        for (auto& [handle, model] : m_uploadedModels)
        {
            m_resourceAllocator.destroyBuffer(model.vertexBuffer);
            m_resourceAllocator.destroyBuffer(model.indexBuffer);
        }
        m_uploadedModels.clear();

        for (auto& [key, texture] : m_uploadedTextures)
        {
            if (texture.view != VK_NULL_HANDLE)
            {
                vkDestroyImageView(
                    m_device.device(),
                    texture.view,
                    nullptr);
                texture.view = VK_NULL_HANDLE;
            }
            m_resourceAllocator.destroyTexture(texture.texture);
        }
        m_uploadedTextures.clear();

        for (auto& [key, sampler] : m_uploadedSamplers)
        {
            if (sampler.sampler != VK_NULL_HANDLE)
            {
                vkDestroySampler(
                    m_device.device(),
                    sampler.sampler,
                    nullptr);
                sampler.sampler = VK_NULL_HANDLE;
            }
        }
        m_uploadedSamplers.clear();

        m_gpuScene.shutdown(m_device.device());

    }

    void VulkanBackend::ensurePathTraceResources()
    {
        const VkExtent2D extent = m_swapchain.extent();
        if (extent.width == 0 || extent.height == 0)
        {
            return;
        }

        if (m_pathTraceResources.accumulation &&
            m_pathTraceResources.tonemap &&
            m_pathTraceResources.width == extent.width &&
            m_pathTraceResources.height == extent.height)
        {
            return;
        }

        destroyPathTraceResources();

        m_pathTraceResources.width = extent.width;
        m_pathTraceResources.height = extent.height;
        m_pathTraceResources.accumulatedSampleCount = 0;
        m_pathTraceResources.resetAccumulation = true;

        m_pathTraceResources.accumulation =
            m_resourceAllocator.createTexture({
                .width = extent.width,
                .height = extent.height,
                .format = TextureFormat::RGBA32_Float,
                .usage =
                    TextureUsageFlags::Storage |
                    TextureUsageFlags::Sampled,
                .memoryUsage = ResourceMemoryUsage::GpuOnly,
                .debugName = "Vulkan path trace accumulation"
            });
        m_pathTraceResources.accumulationLayout =
            VK_IMAGE_LAYOUT_UNDEFINED;
        m_pathTraceResources.accumulationView =
            createTextureView(m_pathTraceResources.accumulation);

        m_pathTraceResources.tonemap =
            m_resourceAllocator.createTexture({
                .width = extent.width,
                .height = extent.height,
                .format = TextureFormat::RGBA8_UNorm,
                .usage =
                    TextureUsageFlags::Storage |
                    TextureUsageFlags::TransferSrc,
                .memoryUsage = ResourceMemoryUsage::GpuOnly,
                .debugName = "Vulkan path trace tonemap"
            });
        m_pathTraceResources.tonemapLayout =
            VK_IMAGE_LAYOUT_UNDEFINED;
        m_pathTraceResources.tonemapView =
            createTextureView(m_pathTraceResources.tonemap);

        const uint32_t frameCount =
            static_cast<uint32_t>(std::max<size_t>(1, m_frameExecutor.framesInFlight()));
        m_pathTraceResources.pathTraceConstants.resize(frameCount);
        m_pathTraceResources.tonemapConstants.resize(frameCount);
        m_pathTraceResources.pathTraceDescriptorSets.assign(
            frameCount,
            VK_NULL_HANDLE);
        m_pathTraceResources.tonemapDescriptorSets.assign(
            frameCount,
            VK_NULL_HANDLE);
        for (uint32_t i = 0; i < frameCount; ++i)
        {
            m_pathTraceResources.pathTraceConstants[i] =
                m_resourceAllocator.createBuffer({
                    .size = sizeof(PathTraceConstants),
                    .usage = BufferUsageFlags::Constant,
                    .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                    .mappedAtCreation = true,
                    .debugName = "Vulkan path trace constants"
                });

            m_pathTraceResources.tonemapConstants[i] =
                m_resourceAllocator.createBuffer({
                    .size = sizeof(TonemapConstants),
                    .usage = BufferUsageFlags::Constant,
                    .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                    .mappedAtCreation = true,
                    .debugName = "Vulkan path trace tonemap constants"
                });
        }

        const uint32_t descriptorSetCapacity = 4u * frameCount;

        VkDescriptorPoolSize poolSizes[5]{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = 4u * frameCount;
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        poolSizes[1].descriptorCount = 4u * frameCount;
        poolSizes[2].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        poolSizes[2].descriptorCount =
            (4u + MaxBindlessTextures) * frameCount;
        poolSizes[3].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[3].descriptorCount = 8u * frameCount;
        poolSizes[4].type = VK_DESCRIPTOR_TYPE_SAMPLER;
        poolSizes[4].descriptorCount =
            (2u + MaxBindlessSamplers) * frameCount;

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = descriptorSetCapacity;
        poolInfo.poolSizeCount =
            static_cast<uint32_t>(std::size(poolSizes));
        poolInfo.pPoolSizes = poolSizes;

        if (vkCreateDescriptorPool(
                m_device.device(),
                &poolInfo,
                nullptr,
                &m_pathTraceResources.descriptorPool) != VK_SUCCESS)
        {
            throw std::runtime_error(
                "Failed to create Vulkan path tracing descriptor pool.");
        }

        ++m_pathTraceResources.pathTraceDescriptorGeneration;
        m_pathTraceResources.pathTraceDescriptorsDirty = true;
        m_pathTraceResources.tonemapDescriptorsDirty = true;
    }

    void VulkanBackend::ensurePathTraceSceneResources(
        const FrameContext& ctx,
        const SceneRenderView& scene)
    {
        if (!ctx.services || !ctx.services->assetManager)
        {
            if (!m_pathTraceResources.sceneVertices)
            {
                uploadPathTraceScene({});
            }
            return;
        }

        AssetManager& assets = *ctx.services->assetManager;
        const bool hasPendingModels =
            std::ranges::any_of(
                scene.models,
                [&assets](const SceneModelRenderItem& item)
                {
                    return !assets.model(item.model);
                });
        const bool retryThisFrame =
            (hasPendingModels ||
                m_pathTraceResources.sceneHadPendingModels ||
                (m_pathTraceResources.sceneTriangleCount == 0u &&
                    !scene.models.empty())) &&
            (m_pathTraceResources.lastSceneBuildFrame == UINT64_MAX ||
                ctx.frameIndex - m_pathTraceResources.lastSceneBuildFrame >=
                    30u);
        if (m_pathTraceResources.sceneVersion == scene.sceneVersion &&
            !retryThisFrame &&
            m_pathTraceResources.sceneVertices)
        {
            return;
        }

        m_pathTraceResources.lastSceneBuildFrame = ctx.frameIndex;

        PathTraceSceneData sceneData =
            buildPathTraceSceneData(
                scene,
                assets,
                [this, &assets](
                    AssetHandle modelHandle,
                    const MaterialTextureSlot& slot)
                {
                    PathTraceMaterialTextureIndices indices{};
                    indices.samplerIndex = requestSampler(nullptr);

                    const ModelAsset* model = assets.model(modelHandle);
                    if (!model ||
                        slot.textureIndex < 0 ||
                        static_cast<size_t>(slot.textureIndex) >=
                            model->textures.size())
                    {
                        return indices;
                    }

                    const TextureAsset& texture =
                        model->textures[static_cast<size_t>(slot.textureIndex)];
                    if (texture.samplerIndex >= 0 &&
                        static_cast<size_t>(texture.samplerIndex) <
                            model->samplers.size())
                    {
                        indices.samplerIndex =
                            requestSampler(
                                &model->samplers[
                                    static_cast<size_t>(texture.samplerIndex)]);
                    }

                    if (texture.imageIndex < 0 ||
                        static_cast<size_t>(texture.imageIndex) >=
                            model->images.size())
                    {
                        return indices;
                    }

                    indices.textureIndex =
                        requestTexture(
                            modelHandle,
                            static_cast<uint32_t>(texture.imageIndex),
                            model->images[static_cast<size_t>(texture.imageIndex)],
                            slot.transfer);
                    return indices;
                });
        if (sceneData.triangles.empty() &&
            m_pathTraceResources.sceneVertices)
        {
            return;
        }

        uploadPathTraceScene(sceneData);

        m_pathTraceResources.sceneVersion = scene.sceneVersion;
        m_pathTraceResources.sceneHadPendingModels = hasPendingModels;
        m_pathTraceResources.accumulatedSampleCount = 0;
        m_pathTraceResources.resetAccumulation = true;
    }

    void VulkanBackend::uploadPathTraceScene(
        const PathTraceSceneData& sceneData)
    {
        retirePathTraceSceneResources();

        m_pathTraceResources.sceneVertexCount =
            static_cast<uint32_t>(sceneData.vertices.size());
        m_pathTraceResources.sceneMaterialCount =
            static_cast<uint32_t>(sceneData.materials.size());
        m_pathTraceResources.sceneTriangleCount =
            static_cast<uint32_t>(sceneData.triangles.size());
        m_pathTraceResources.sceneBvhNodeCount =
            static_cast<uint32_t>(sceneData.bvhNodes.size());
        m_pathTraceResources.firstEmissiveTriangleIndex =
            sceneData.firstEmissiveTriangleIndex;

        spdlog::info(
            "[Vulkan] Path trace scene upload: vertices={} materials={} triangles={} bvhNodes={} firstEmissiveTriangle={}",
            m_pathTraceResources.sceneVertexCount,
            m_pathTraceResources.sceneMaterialCount,
            m_pathTraceResources.sceneTriangleCount,
            m_pathTraceResources.sceneBvhNodeCount,
            m_pathTraceResources.firstEmissiveTriangleIndex);

        struct PendingSceneUpload
        {
            VulkanBuffer* destination = nullptr;
            VulkanBuffer staging;
            VkDeviceSize byteSize = 0;
        };

        std::vector<PendingSceneUpload> pendingUploads;
        pendingUploads.reserve(4);

        auto createBuffer =
            [&](VkDeviceSize elementSize,
                uint32_t elementCount,
                const void* data,
                const char* debugName)
            {
                const VkDeviceSize byteSize =
                    elementSize * std::max<uint32_t>(1u, elementCount);

                VulkanBuffer buffer =
                    m_resourceAllocator.createBuffer({
                        .size = byteSize,
                        .usage =
                            BufferUsageFlags::Storage |
                            BufferUsageFlags::TransferDst,
                        .memoryUsage = ResourceMemoryUsage::GpuOnly,
                        .mappedAtCreation = false,
                        .debugName = debugName
                    });

                if (elementCount != 0u && data)
                {
                    VulkanBuffer staging =
                        m_resourceAllocator.createBuffer({
                            .size = byteSize,
                            .usage = BufferUsageFlags::TransferSrc,
                            .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                            .mappedAtCreation = true,
                            .debugName = "Vulkan path trace scene staging"
                        });

                    std::memcpy(
                        staging.mapped,
                        data,
                        static_cast<size_t>(elementSize * elementCount));
                    m_resourceAllocator.flush(
                        staging,
                        0,
                        elementSize * elementCount);

                    pendingUploads.push_back({
                        .destination = nullptr,
                        .staging = std::move(staging),
                        .byteSize = elementSize * elementCount
                    });
                }

                return buffer;
            };

        m_pathTraceResources.sceneVertices =
            createBuffer(
                sizeof(PathTraceVertex),
                m_pathTraceResources.sceneVertexCount,
                sceneData.vertices.data(),
                "Vulkan path trace scene vertices");
        m_pathTraceResources.sceneMaterials =
            createBuffer(
                sizeof(PathTraceMaterial),
                m_pathTraceResources.sceneMaterialCount,
                sceneData.materials.data(),
                "Vulkan path trace scene materials");
        m_pathTraceResources.sceneTriangles =
            createBuffer(
                sizeof(PathTraceTriangle),
                m_pathTraceResources.sceneTriangleCount,
                sceneData.triangles.data(),
                "Vulkan path trace scene triangles");
        m_pathTraceResources.sceneBvhNodes =
            createBuffer(
                sizeof(PathTraceBVHNode),
                m_pathTraceResources.sceneBvhNodeCount,
                sceneData.bvhNodes.data(),
                "Vulkan path trace scene BVH nodes");

        uint32_t uploadIndex = 0;
        auto bindPendingUpload =
            [&](VulkanBuffer& destination, uint32_t elementCount)
            {
                if (elementCount == 0u)
                {
                    return;
                }

                pendingUploads[uploadIndex++].destination = &destination;
            };

        bindPendingUpload(
            m_pathTraceResources.sceneVertices,
            m_pathTraceResources.sceneVertexCount);
        bindPendingUpload(
            m_pathTraceResources.sceneMaterials,
            m_pathTraceResources.sceneMaterialCount);
        bindPendingUpload(
            m_pathTraceResources.sceneTriangles,
            m_pathTraceResources.sceneTriangleCount);
        bindPendingUpload(
            m_pathTraceResources.sceneBvhNodes,
            m_pathTraceResources.sceneBvhNodeCount);

        if (!pendingUploads.empty())
        {
            m_uploadScheduler.record(
                [&](VkCommandBuffer cmd)
                {
                    for (const PendingSceneUpload& upload : pendingUploads)
                    {
                        VkBufferCopy copy{};
                        copy.size = upload.byteSize;

                        vkCmdCopyBuffer(
                            cmd,
                            upload.staging.buffer,
                            upload.destination->buffer,
                            1,
                            &copy);

                        VkBufferMemoryBarrier barrier{};
                        barrier.sType =
                            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                        barrier.srcAccessMask =
                            VK_ACCESS_TRANSFER_WRITE_BIT;
                        barrier.dstAccessMask = 0;
                        barrier.srcQueueFamilyIndex =
                            VK_QUEUE_FAMILY_IGNORED;
                        barrier.dstQueueFamilyIndex =
                            VK_QUEUE_FAMILY_IGNORED;
                        barrier.buffer = upload.destination->buffer;
                        barrier.offset = 0;
                        barrier.size = upload.byteSize;

                        vkCmdPipelineBarrier(
                            cmd,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            0,
                            0,
                            nullptr,
                            1,
                            &barrier,
                            0,
                            nullptr);
                    }
                });

            for (PendingSceneUpload& upload : pendingUploads)
            {
                m_retirementQueue.retire(std::move(upload.staging));
            }
        }

        ++m_pathTraceResources.pathTraceDescriptorGeneration;
        m_pathTraceResources.pathTraceDescriptorsDirty = true;
    }

    void VulkanBackend::retirePathTraceSceneResources()
    {
        m_retirementQueue.retire(
            std::move(m_pathTraceResources.sceneVertices));
        m_retirementQueue.retire(
            std::move(m_pathTraceResources.sceneMaterials));
        m_retirementQueue.retire(
            std::move(m_pathTraceResources.sceneTriangles));
        m_retirementQueue.retire(
            std::move(m_pathTraceResources.sceneBvhNodes));
    }

    void VulkanBackend::destroyPathTraceResources()
    {
        m_retirementQueue.retireImageView(
            m_pathTraceResources.accumulationView);
        m_retirementQueue.retireImageView(m_pathTraceResources.tonemapView);
        m_retirementQueue.retireDescriptorPool(
            m_pathTraceResources.descriptorPool);
        retirePathTraceSceneResources();
        m_retirementQueue.retire(
            std::move(m_pathTraceResources.accumulation));
        m_retirementQueue.retire(std::move(m_pathTraceResources.tonemap));
        for (VulkanBuffer& buffer : m_pathTraceResources.pathTraceConstants)
        {
            m_retirementQueue.retire(std::move(buffer));
        }
        for (VulkanBuffer& buffer : m_pathTraceResources.tonemapConstants)
        {
            m_retirementQueue.retire(std::move(buffer));
        }

        m_pathTraceResources = {};
    }



    VkImageView VulkanBackend::createTextureView(
        const VulkanTexture& texture) const
    {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = texture.image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = texture.format;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.layerCount = 1;

        VkImageView view = VK_NULL_HANDLE;
        if (vkCreateImageView(
                m_device.device(),
                &viewInfo,
                nullptr,
                &view) != VK_SUCCESS)
        {
            throw std::runtime_error(
                "Failed to create Vulkan path tracing texture view.");
        }

        return view;
    }

    void VulkanBackend::updatePathTraceDescriptors(
        const VulkanComputePipeline* pathTracePipeline,
        const VulkanComputePipeline* tonemapPipeline,
        uint32_t frameSlot)
    {
        if (m_pathTraceResources.descriptorPool == VK_NULL_HANDLE)
        {
            return;
        }

        if (!m_pathTraceResources.accumulation ||
            m_pathTraceResources.pathTraceConstants.empty() ||
            m_pathTraceResources.pathTraceDescriptorSets.empty())
        {
            return;
        }

        if (pathTracePipeline)
        {
            bool updateDescriptors =
                m_pathTraceResources.pathTraceDescriptorsDirty ||
                frameSlot >= m_pathTraceResources.pathTraceDescriptorVersions.size() ||
                m_pathTraceResources.pathTraceDescriptorVersions[frameSlot] !=
                    m_pathTraceResources.pathTraceDescriptorGeneration;

            const bool needAllocate =
                m_pathTraceResources.pathTraceDescriptorSets.empty() ||
                std::ranges::any_of(
                    m_pathTraceResources.pathTraceDescriptorSets,
                    [](VkDescriptorSet set)
                    {
                        return set == VK_NULL_HANDLE;
                    });

            if (needAllocate)
            {
                if (tonemapPipeline == nullptr)
                {
                    vkResetDescriptorPool(
                        m_device.device(),
                        m_pathTraceResources.descriptorPool,
                        0);
                    std::ranges::fill(
                        m_pathTraceResources.pathTraceDescriptorSets,
                        VK_NULL_HANDLE);
                    std::ranges::fill(
                        m_pathTraceResources.tonemapDescriptorSets,
                        VK_NULL_HANDLE);
                    ++m_pathTraceResources.pathTraceDescriptorGeneration;
                    m_pathTraceResources.pathTraceDescriptorsDirty = true;
                    m_pathTraceResources.tonemapDescriptorsDirty = true;
                    updateDescriptors = true;
                }

                const uint32_t setCount =
                    static_cast<uint32_t>(
                        m_pathTraceResources.pathTraceDescriptorSets.size());
                if (setCount == 0u)
                {
                    return;
                }

                std::vector<VkDescriptorSetLayout> layouts(
                    setCount,
                    pathTracePipeline->descriptorSetLayout);

                VkDescriptorSetAllocateInfo allocateInfo{};
                allocateInfo.sType =
                    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                allocateInfo.descriptorPool =
                    m_pathTraceResources.descriptorPool;
                allocateInfo.descriptorSetCount = setCount;
                allocateInfo.pSetLayouts = layouts.data();

                const VkResult result = vkAllocateDescriptorSets(
                        m_device.device(),
                        &allocateInfo,
                        m_pathTraceResources.pathTraceDescriptorSets.data());
                if (result != VK_SUCCESS)
                {
                    spdlog::error(
                        "Failed to allocate {} Vulkan path trace descriptor sets from pool (result={})",
                        setCount,
                        static_cast<int32_t>(result));
                    throw std::runtime_error(
                        "Failed to allocate Vulkan path trace descriptor set.");
                }

                updateDescriptors = true;
                m_pathTraceResources.pathTraceDescriptorVersions.assign(
                    setCount, 0);
            }

            if (!updateDescriptors)
            {
                return;
            }

            VkDescriptorBufferInfo vertexInfo{};
            vertexInfo.buffer = m_pathTraceResources.sceneVertices.buffer;
            vertexInfo.range = m_pathTraceResources.sceneVertices.size;

            VkDescriptorBufferInfo materialInfo{};
            materialInfo.buffer = m_pathTraceResources.sceneMaterials.buffer;
            materialInfo.range = m_pathTraceResources.sceneMaterials.size;

            VkDescriptorBufferInfo triangleInfo{};
            triangleInfo.buffer = m_pathTraceResources.sceneTriangles.buffer;
            triangleInfo.range = m_pathTraceResources.sceneTriangles.size;

            VkDescriptorBufferInfo bvhInfo{};
            bvhInfo.buffer = m_pathTraceResources.sceneBvhNodes.buffer;
            bvhInfo.range = m_pathTraceResources.sceneBvhNodes.size;

            VkDescriptorImageInfo accumulationInfo{};
            accumulationInfo.imageView =
                m_pathTraceResources.accumulationView;
            accumulationInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkDescriptorImageInfo environmentInfo{};
            environmentInfo.imageView =
                m_environmentResources.cubemapView;
            environmentInfo.imageLayout =
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorImageInfo environmentSamplerInfo{};
            environmentSamplerInfo.sampler =
                m_environmentResources.sampler;

            VkDescriptorBufferInfo* sceneInfos[4] =
            {
                &vertexInfo,
                &materialInfo,
                &triangleInfo,
                &bvhInfo
            };

            std::vector<VkDescriptorImageInfo> textureInfos(
                m_uploadedTextures.size());
            for (const auto& [key, texture] : m_uploadedTextures)
            {
                if (texture.descriptorIndex >= textureInfos.size())
                {
                    continue;
                }

                textureInfos[texture.descriptorIndex].imageView = texture.view;
                textureInfos[texture.descriptorIndex].imageLayout =
                    texture.layout;
            }

            std::vector<VkDescriptorImageInfo> samplerInfos(
                m_uploadedSamplers.size());
            for (const auto& [key, sampler] : m_uploadedSamplers)
            {
                if (sampler.descriptorIndex >= samplerInfos.size())
                {
                    continue;
                }

                samplerInfos[sampler.descriptorIndex].sampler =
                    sampler.sampler;
            }

            std::vector<VkDescriptorBufferInfo> constantsInfos(
                m_pathTraceResources.pathTraceConstants.size());
            std::vector<VkWriteDescriptorSet> writes;
            writes.reserve(10u);

            const size_t setIndex = frameSlot;
            if (setIndex < m_pathTraceResources.pathTraceDescriptorSets.size())
            {
                constantsInfos[setIndex].buffer =
                    m_pathTraceResources.pathTraceConstants[setIndex].buffer;
                constantsInfos[setIndex].range = sizeof(PathTraceConstants);

                VkWriteDescriptorSet constantsWrite{};
                constantsWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                constantsWrite.dstSet =
                    m_pathTraceResources.pathTraceDescriptorSets[setIndex];
                constantsWrite.dstBinding = 0;
                constantsWrite.descriptorCount = 1;
                constantsWrite.descriptorType =
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                constantsWrite.pBufferInfo = &constantsInfos[setIndex];
                writes.push_back(constantsWrite);

                VkWriteDescriptorSet accumulationWrite{};
                accumulationWrite.sType =
                    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                accumulationWrite.dstSet =
                    m_pathTraceResources.pathTraceDescriptorSets[setIndex];
                accumulationWrite.dstBinding = 1;
                accumulationWrite.descriptorCount = 1;
                accumulationWrite.descriptorType =
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                accumulationWrite.pImageInfo = &accumulationInfo;
                writes.push_back(accumulationWrite);

                for (uint32_t i = 0; i < 4; ++i)
                {
                    VkWriteDescriptorSet sceneWrite{};
                    sceneWrite.sType =
                        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    sceneWrite.dstSet =
                        m_pathTraceResources.pathTraceDescriptorSets[setIndex];
                    sceneWrite.dstBinding = 2 + i;
                    sceneWrite.descriptorCount = 1;
                    sceneWrite.descriptorType =
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    sceneWrite.pBufferInfo = sceneInfos[i];
                    writes.push_back(sceneWrite);
                }

                if (m_environmentResources.cubemapView != VK_NULL_HANDLE &&
                    m_environmentResources.sampler != VK_NULL_HANDLE)
                {
                    VkWriteDescriptorSet environmentWrite{};
                    environmentWrite.sType =
                        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    environmentWrite.dstSet =
                        m_pathTraceResources.pathTraceDescriptorSets[setIndex];
                    environmentWrite.dstBinding = 6;
                    environmentWrite.descriptorCount = 1;
                    environmentWrite.descriptorType =
                        VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                    environmentWrite.pImageInfo = &environmentInfo;
                    writes.push_back(environmentWrite);

                    VkWriteDescriptorSet samplerWrite{};
                    samplerWrite.sType =
                        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    samplerWrite.dstSet =
                        m_pathTraceResources.pathTraceDescriptorSets[setIndex];
                    samplerWrite.dstBinding = 7;
                    samplerWrite.descriptorCount = 1;
                    samplerWrite.descriptorType =
                        VK_DESCRIPTOR_TYPE_SAMPLER;
                    samplerWrite.pImageInfo = &environmentSamplerInfo;
                    writes.push_back(samplerWrite);
                }

                if (!textureInfos.empty())
                {
                    VkWriteDescriptorSet texturesWrite{};
                    texturesWrite.sType =
                        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    texturesWrite.dstSet =
                        m_pathTraceResources.pathTraceDescriptorSets[setIndex];
                    texturesWrite.dstBinding = MaxBindlessTextures + 2u;
                    texturesWrite.descriptorCount =
                        static_cast<uint32_t>(textureInfos.size());
                    texturesWrite.descriptorType =
                        VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                    texturesWrite.pImageInfo = textureInfos.data();
                    writes.push_back(texturesWrite);
                }

                if (!samplerInfos.empty())
                {
                    VkWriteDescriptorSet samplersWrite{};
                    samplersWrite.sType =
                        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    samplersWrite.dstSet =
                        m_pathTraceResources.pathTraceDescriptorSets[setIndex];
                    samplersWrite.dstBinding = MaxBindlessSamplers;
                    samplersWrite.descriptorCount =
                        static_cast<uint32_t>(samplerInfos.size());
                    samplersWrite.descriptorType =
                        VK_DESCRIPTOR_TYPE_SAMPLER;
                    samplersWrite.pImageInfo = samplerInfos.data();
                    writes.push_back(samplersWrite);
                }
            }

            vkUpdateDescriptorSets(
                m_device.device(),
                static_cast<uint32_t>(writes.size()),
                writes.data(),
                0,
                nullptr);

            m_pathTraceResources.pathTraceDescriptorVersions[frameSlot] =
                m_pathTraceResources.pathTraceDescriptorGeneration;
            m_pathTraceResources.pathTraceDescriptorsDirty =
                std::ranges::any_of(
                    m_pathTraceResources.pathTraceDescriptorVersions,
                    [this](uint64_t version)
                    {
                        return version != m_pathTraceResources
                            .pathTraceDescriptorGeneration;
                    });
        }

        if (tonemapPipeline)
        {
            bool updateDescriptors =
                m_pathTraceResources.tonemapDescriptorsDirty;

            const bool needAllocate =
                m_pathTraceResources.tonemapDescriptorSets.empty() ||
                std::ranges::any_of(
                    m_pathTraceResources.tonemapDescriptorSets,
                    [](VkDescriptorSet set)
                    {
                        return set == VK_NULL_HANDLE;
                    });

            if (needAllocate)
            {
                const uint32_t setCount =
                    static_cast<uint32_t>(
                        m_pathTraceResources.tonemapDescriptorSets.size());
                std::vector<VkDescriptorSetLayout> layouts(
                    setCount,
                    tonemapPipeline->descriptorSetLayout);

                VkDescriptorSetAllocateInfo allocateInfo{};
                allocateInfo.sType =
                    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                allocateInfo.descriptorPool =
                    m_pathTraceResources.descriptorPool;
                allocateInfo.descriptorSetCount = setCount;
                allocateInfo.pSetLayouts = layouts.data();

                const VkResult result = vkAllocateDescriptorSets(
                        m_device.device(),
                        &allocateInfo,
                        m_pathTraceResources.tonemapDescriptorSets.data());
                if (result != VK_SUCCESS)
                {
                    spdlog::error(
                        "Failed to allocate {} Vulkan path trace tonemap descriptor sets from pool (result={})",
                        setCount,
                        static_cast<int32_t>(result));
                    throw std::runtime_error(
                        "Failed to allocate Vulkan path trace tonemap descriptor set.");
                }

                updateDescriptors = true;
            }

            if (!updateDescriptors)
            {
                return;
            }

            VkDescriptorImageInfo tonemapInfo{};
            tonemapInfo.imageView = m_pathTraceResources.tonemapView;
            tonemapInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkDescriptorImageInfo accumulationInfo{};
            accumulationInfo.imageView =
                m_pathTraceResources.accumulationView;
            accumulationInfo.imageLayout =
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            std::vector<VkDescriptorBufferInfo> constantsInfos(
                m_pathTraceResources.tonemapConstants.size());
            std::vector<VkWriteDescriptorSet> writes;
            writes.reserve(
                m_pathTraceResources.tonemapDescriptorSets.size() * 3u);

            for (size_t setIndex = 0;
                 setIndex < m_pathTraceResources.tonemapDescriptorSets.size();
                 ++setIndex)
            {
                constantsInfos[setIndex].buffer =
                    m_pathTraceResources.tonemapConstants[setIndex].buffer;
                constantsInfos[setIndex].range = sizeof(TonemapConstants);

                VkWriteDescriptorSet constantsWrite{};
                constantsWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                constantsWrite.dstSet =
                    m_pathTraceResources.tonemapDescriptorSets[setIndex];
                constantsWrite.dstBinding = 0;
                constantsWrite.descriptorCount = 1;
                constantsWrite.descriptorType =
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                constantsWrite.pBufferInfo = &constantsInfos[setIndex];
                writes.push_back(constantsWrite);

                VkWriteDescriptorSet tonemapWrite{};
                tonemapWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                tonemapWrite.dstSet =
                    m_pathTraceResources.tonemapDescriptorSets[setIndex];
                tonemapWrite.dstBinding = 1;
                tonemapWrite.descriptorCount = 1;
                tonemapWrite.descriptorType =
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                tonemapWrite.pImageInfo = &tonemapInfo;
                writes.push_back(tonemapWrite);

                VkWriteDescriptorSet accumulationWrite{};
                accumulationWrite.sType =
                    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                accumulationWrite.dstSet =
                    m_pathTraceResources.tonemapDescriptorSets[setIndex];
                accumulationWrite.dstBinding = 2;
                accumulationWrite.descriptorCount = 1;
                accumulationWrite.descriptorType =
                    VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                accumulationWrite.pImageInfo = &accumulationInfo;
                writes.push_back(accumulationWrite);
            }

            vkUpdateDescriptorSets(
                m_device.device(),
                static_cast<uint32_t>(writes.size()),
                writes.data(),
                0,
                nullptr);

            m_pathTraceResources.tonemapDescriptorsDirty = false;
        }
    }

    void VulkanBackend::ensureComputeTestResources(
        const VulkanComputePipeline& pipeline)
    {
        if (m_computeTestBuffer &&
            m_computeTestDescriptorSet != VK_NULL_HANDLE)
        {
            return;
        }

        if (!m_computeTestBuffer)
        {
            m_computeTestBuffer =
                m_resourceAllocator.createBuffer({
                    .size = 64 * 64 * sizeof(uint32_t),
                    .usage = BufferUsageFlags::Storage,
                    .memoryUsage = ResourceMemoryUsage::GpuOnly,
                    .debugName = "Vulkan compute binding test buffer"
                });
        }

        if (m_computeTestDescriptorPool == VK_NULL_HANDLE)
        {
            VkDescriptorPoolSize poolSize{};
            poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            poolSize.descriptorCount = 1;

            VkDescriptorPoolCreateInfo poolInfo{};
            poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.maxSets = 1;
            poolInfo.poolSizeCount = 1;
            poolInfo.pPoolSizes = &poolSize;

            if (vkCreateDescriptorPool(
                    m_device.device(),
                    &poolInfo,
                    nullptr,
                    &m_computeTestDescriptorPool) != VK_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to create Vulkan compute test descriptor pool.");
            }
        }

        VkDescriptorSetAllocateInfo allocateInfo{};
        allocateInfo.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocateInfo.descriptorPool = m_computeTestDescriptorPool;
        allocateInfo.descriptorSetCount = 1;
        allocateInfo.pSetLayouts = &pipeline.descriptorSetLayout;

        if (vkAllocateDescriptorSets(
                m_device.device(),
                &allocateInfo,
                &m_computeTestDescriptorSet) != VK_SUCCESS)
        {
            throw std::runtime_error(
                "Failed to allocate Vulkan compute test descriptor set.");
        }

        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = m_computeTestBuffer.buffer;
        bufferInfo.offset = 0;
        bufferInfo.range = m_computeTestBuffer.size;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = m_computeTestDescriptorSet;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(
            m_device.device(),
            1,
            &write,
            0,
            nullptr);
    }


    void VulkanBackend::ensureClusteredForwardResources()
    {
        const VkExtent2D extent = m_swapchain.extent();
        if (extent.width == 0 || extent.height == 0)
        {
            return;
        }

        const uint32_t clusterCountX =
            (extent.width + ClusteredForwardTileSizeX - 1u) /
            ClusteredForwardTileSizeX;
        const uint32_t clusterCountY =
            (extent.height + ClusteredForwardTileSizeY - 1u) /
            ClusteredForwardTileSizeY;
        const uint32_t clusterCountZ = ClusteredForwardSliceCountZ;
        const uint32_t clusterCount =
            clusterCountX * clusterCountY * clusterCountZ;
        const uint32_t hiZMipCount =
            1u + static_cast<uint32_t>(
                std::floor(std::log2(std::max(extent.width, extent.height))));
        const uint32_t maxInstances = ClusteredForwardMaxGpuCullInstances;

        // clusterCount is the "initialized" marker (there is no backend-owned
            // cluster buffer any more because they are graph-registry owned).
        if (m_clusteredForwardResources.clusterCount != 0 &&
            m_clusteredForwardResources.width == extent.width &&
            m_clusteredForwardResources.height == extent.height)
        {
            return;
        }

        destroyClusteredForwardResources();

        m_clusteredForwardResources.width = extent.width;
        m_clusteredForwardResources.height = extent.height;
        m_clusteredForwardResources.clusterCountX = clusterCountX;
        m_clusteredForwardResources.clusterCountY = clusterCountY;
        m_clusteredForwardResources.clusterCountZ = clusterCountZ;
        m_clusteredForwardResources.clusterCount = clusterCount;
        m_clusteredForwardResources.hiZMipCount = hiZMipCount;

        spdlog::info(
            "[VulkanBackend] Clustered forward resources: {}x{}x{} clusters, Hi-Z {}x{} mips={}, maxCullInstances={}",
            clusterCountX,
            clusterCountY,
            clusterCountZ,
            extent.width,
            extent.height,
            hiZMipCount,
            maxInstances);
    }

    void VulkanBackend::resolveGpuDrivenBuffers(const CompiledGraphPlan& plan)
    {
        m_gpuDrivenBuffers = GpuDrivenBuffers{};

        auto entryFor =
            [&](GraphResourceSemantic semantic) -> VulkanGraphResourceEntry*
            {
                const GraphResourceId id =
                    findResourceBySemantic(plan, semantic);
                return id != InvalidGraphResourceId
                    ? m_graphResourceRegistry.entry(id)
                    : nullptr;
            };
        auto bind =
            [&](VkBuffer& buffer, VkDeviceSize& size,
                GraphResourceSemantic semantic) -> VulkanGraphResourceEntry*
            {
                VulkanGraphResourceEntry* entry = entryFor(semantic);
                if (entry && entry->buffer)
                {
                    buffer = entry->buffer.buffer;
                    size = entry->buffer.size;
                }
                return entry;
            };

        m_gpuDrivenBuffers.instanceBoundsEntry = bind(
            m_gpuDrivenBuffers.instanceBounds,
            m_gpuDrivenBuffers.instanceBoundsSize,
            GraphResourceSemantic::GpuDrivenInstanceBounds);
        m_gpuDrivenBuffers.drawInputsEntry = bind(
            m_gpuDrivenBuffers.drawInputs,
            m_gpuDrivenBuffers.drawInputsSize,
            GraphResourceSemantic::GpuDrivenDrawInputs);
        bind(
            m_gpuDrivenBuffers.visibleInstances,
            m_gpuDrivenBuffers.visibleInstancesSize,
            GraphResourceSemantic::GpuDrivenVisibleInstances);
        bind(
            m_gpuDrivenBuffers.visibleCount,
            m_gpuDrivenBuffers.visibleCountSize,
            GraphResourceSemantic::GpuDrivenVisibleCount);
        bind(
            m_gpuDrivenBuffers.indirectArguments,
            m_gpuDrivenBuffers.indirectArgumentsSize,
            GraphResourceSemantic::GpuDrivenIndirectArguments);
        bind(
            m_gpuDrivenBuffers.drawMetadata,
            m_gpuDrivenBuffers.drawMetadataSize,
            GraphResourceSemantic::GpuDrivenDrawMetadata);
        bind(
            m_gpuDrivenBuffers.binCounts,
            m_gpuDrivenBuffers.binCountsSize,
            GraphResourceSemantic::GpuDrivenBinCounts);
    }

    void VulkanBackend::resolveClusterBuffers(const CompiledGraphPlan& plan)
    {
        m_clusterBuffers = ClusterBuffers{};

        auto bind =
            [&](VkBuffer& buffer, VkDeviceSize& size,
                GraphResourceSemantic semantic) -> VulkanGraphResourceEntry*
            {
                const GraphResourceId id =
                    findResourceBySemantic(plan, semantic);
                VulkanGraphResourceEntry* entry =
                    id != InvalidGraphResourceId
                        ? m_graphResourceRegistry.entry(id)
                        : nullptr;
                if (entry && entry->buffer)
                {
                    buffer = entry->buffer.buffer;
                    size = entry->buffer.size;
                }
                return entry;
            };

        bind(m_clusterBuffers.bounds, m_clusterBuffers.boundsSize,
            GraphResourceSemantic::ClusterBounds);
        bind(m_clusterBuffers.lightGrid, m_clusterBuffers.lightGridSize,
            GraphResourceSemantic::ClusterLightGrid);
        bind(m_clusterBuffers.lightIndices, m_clusterBuffers.lightIndicesSize,
            GraphResourceSemantic::ClusterLightIndices);
        bind(m_clusterBuffers.lightCounter, m_clusterBuffers.lightCounterSize,
            GraphResourceSemantic::ClusterLightCounter);
        m_clusterBuffers.visibleLightsEntry = bind(
            m_clusterBuffers.visibleLights, m_clusterBuffers.visibleLightsSize,
            GraphResourceSemantic::VisibleLights);
    }

    void VulkanBackend::uploadGpuDrivenInputs()
    {
        // Upload this frame's cull inputs straight into the graph-owned mapped
        // buffers, then flush (host-visible memory may be non-coherent).
        VulkanGraphResourceEntry* boundsEntry =
            m_gpuDrivenBuffers.instanceBoundsEntry;
        VulkanGraphResourceEntry* inputsEntry =
            m_gpuDrivenBuffers.drawInputsEntry;
        if (boundsEntry && boundsEntry->buffer.mapped)
        {
            const std::span<const GpuInstanceBounds> bounds =
                m_gpuScene.preparedInstanceBounds();
            const size_t bytes = std::min<size_t>(
                bounds.size() * sizeof(GpuInstanceBounds),
                static_cast<size_t>(boundsEntry->buffer.size));
            if (bytes > 0)
            {
                std::memcpy(boundsEntry->buffer.mapped, bounds.data(), bytes);
                m_resourceAllocator.flush(boundsEntry->buffer, 0, bytes);
            }
        }
        if (inputsEntry && inputsEntry->buffer.mapped)
        {
            const std::span<const GpuDrawInput> inputs =
                m_gpuScene.preparedDrawInputs();
            const size_t bytes = std::min<size_t>(
                inputs.size() * sizeof(GpuDrawInput),
                static_cast<size_t>(inputsEntry->buffer.size));
            if (bytes > 0)
            {
                std::memcpy(inputsEntry->buffer.mapped, inputs.data(), bytes);
                m_resourceAllocator.flush(inputsEntry->buffer, 0, bytes);
            }
        }

        VulkanGraphResourceEntry* lightsEntry =
            m_clusterBuffers.visibleLightsEntry;
        if (lightsEntry && lightsEntry->buffer.mapped)
        {
            const std::span<const GpuVisibleLight> lights =
                m_gpuScene.preparedVisibleLights();
            const size_t bytes = std::min<size_t>(
                lights.size() * sizeof(GpuVisibleLight),
                static_cast<size_t>(lightsEntry->buffer.size));
            if (bytes > 0)
            {
                std::memcpy(lightsEntry->buffer.mapped, lights.data(), bytes);
                m_resourceAllocator.flush(lightsEntry->buffer, 0, bytes);
            }
        }
    }

    bool VulkanBackend::bindClusteredForwardCompute(
        const VulkanComputePipeline& pipeline,
        const FrameContext& ctx,
        const SceneRenderView& scene,
        VkCommandBuffer cmd)
    {
        const PipelineId graphicsPipelineId =
            makePipelineId("clustered_forward_opaque");
        const GraphicsPipelineHandle graphicsHandle = m_pipelineLibrary
            ? m_pipelineManager.resolveGraphicsPipeline(
                *m_pipelineLibrary,
                graphicsPipelineId,
                swapchainTextureFormat())
            : GraphicsPipelineHandle{};

        if (!graphicsHandle ||
            !prepareSceneResources(ctx, scene, graphicsHandle))
        {
            return false;
        }

        const uint32_t frameSlot =
            static_cast<uint32_t>(
                ctx.frameIndex % m_gpuScene.frameSlotCount());
        VkDescriptorSet descriptorSet =
            m_gpuScene.frameResources(frameSlot).descriptorSet;
        if (descriptorSet == VK_NULL_HANDLE)
        {
            return false;
        }

        VulkanPassContext passCtx{};
        passCtx.cmd = cmd;
        recordClusteredForwardCompute(passCtx, pipeline, descriptorSet);
        return true;
    }

    void VulkanBackend::destroyClusteredForwardResources()
    {
        // Cluster buffers are graph-registry owned; nothing to free here.
        destroyHiZDebugDescriptors();
        if (m_clusteredForwardResources.hiZDebugSampler != VK_NULL_HANDLE)
        {
            vkDestroySampler(
                m_device.device(),
                m_clusteredForwardResources.hiZDebugSampler,
                nullptr);
        }
        m_clusteredForwardResources = {};
    }

    void VulkanBackend::ensureDepthTarget()
    {
        const VkExtent2D extent = m_swapchain.extent();
        if (m_depthTexture &&
            m_depthWidth == extent.width &&
            m_depthHeight == extent.height)
        {
            return;
        }

        destroyDepthTarget();

        m_depthTexture =
            m_resourceAllocator.createTexture({
                .width = extent.width,
                .height = extent.height,
                .format = TextureFormat::D32_Float,
                .usage =
                    TextureUsageFlags::DepthAttachment |
                    TextureUsageFlags::Sampled,
                .memoryUsage = ResourceMemoryUsage::GpuOnly,
                .debugName = "Vulkan forward depth"
            });

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = m_depthTexture.image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_D32_SFLOAT;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(
                m_device.device(),
                &viewInfo,
                nullptr,
                &m_depthImageView) != VK_SUCCESS)
        {
            throw std::runtime_error(
                "Failed to create Vulkan depth image view.");
        }

        m_depthLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        m_depthWidth = extent.width;
        m_depthHeight = extent.height;
    }

    void VulkanBackend::destroyDepthTarget()
    {
        if (m_depthImageView != VK_NULL_HANDLE)
        {
            vkDestroyImageView(
                m_device.device(),
                m_depthImageView,
                nullptr);
            m_depthImageView = VK_NULL_HANDLE;
        }

        m_resourceAllocator.destroyTexture(m_depthTexture);
        m_depthLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        m_depthWidth = 0;
        m_depthHeight = 0;
    }

    std::vector<IBLBakeResult> VulkanBackend::executeIBLBakeRequests(
        std::span<const IBLBakeRequest> requests,
        const FrameContext& ctx)
    {
        std::vector<IBLBakeResult> results;
        results.reserve(requests.size());

        for (const IBLBakeRequest& request : requests)
        {
            IBLBakeResult result{};
            result.requestId = request.requestId;
            result.handle = request.handle;

            if (!ctx.services || !ctx.services->assetManager)
            {
                result.error = "SourceNotReady";
                results.push_back(std::move(result));
                continue;
            }

            const ImageAsset* image =
                ctx.services->assetManager->image(
                    request.desc.sourceEnvironment);
            if (!image || !image->valid())
            {
                result.error = "SourceNotReady";
                results.push_back(std::move(result));
                continue;
            }

            auto computePipeline = [&](const char* name) -> VulkanComputePipeline*
                {
                    const PipelineId pipelineId = makePipelineId(name);
                    const ComputePipelineHandle handle =
                        m_pipelineManager.resolveComputePipeline(
                            *m_pipelineLibrary, pipelineId);
                    return m_pipelineManager.computePipeline(handle);
                };

            VulkanComputePipeline* convertPipeline =
                computePipeline("equirect_to_cubemap");
            VulkanComputePipeline* irradiancePipeline =
                computePipeline("ibl_irradiance");
            VulkanComputePipeline* prefilterPipeline =
                computePipeline("ibl_prefilter");
            VulkanComputePipeline* brdfPipeline =
                computePipeline("ibl_brdf_lut");

            if (!convertPipeline ||
                !irradiancePipeline ||
                !prefilterPipeline ||
                !brdfPipeline)
            {
                result.error = "Missing Vulkan IBL compute pipeline";
                results.push_back(std::move(result));
                continue;
            }

            try
            {
                (void)requestTexture(
                    request.desc.sourceEnvironment,
                    0,
                    *image,
                    TextureTransferFunction::Linear,
                    false);

                SceneRenderView bakeScene{};
                bakeScene.environment.equirectTexture =
                    request.desc.sourceEnvironment;
                bakeScene.environment.settings.cubemapSize =
                    std::max(1u, request.desc.environmentSize);
                bakeScene.environment.settings.intensity = 1.0f;
                bakeScene.environment.version = request.requestId;
                bakeScene.environment.enabled = 1u;

                m_environmentResources.irradianceSize =
                    std::max(1u, request.desc.irradianceSize);
                m_environmentResources.prefilterSize =
                    std::max(1u, request.desc.prefilterSize);
                m_environmentResources.prefilterMipCount =
                    mipCountForSize(m_environmentResources.prefilterSize);
                m_environmentResources.brdfLutSize =
                    std::max(1u, request.desc.brdfLutSize);

                m_commandSystem.immediateSubmit(
                    m_device.graphicsQueue(),
                    [&](VkCommandBuffer cmd)
                    {
                        (void)ensureEnvironmentResources(
                            ctx,
                            bakeScene,
                            cmd);
                        (void)convertEnvironmentIfReady(
                            *convertPipeline,
                            ctx,
                            bakeScene,
                            cmd);

                        if (m_environmentResources.bakeDescriptorPool ==
                            VK_NULL_HANDLE)
                        {
                            VkDescriptorPoolSize bakePoolSizes[3]{};
                            bakePoolSizes[0].type =
                                VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                            bakePoolSizes[0].descriptorCount = 64;
                            bakePoolSizes[1].type =
                                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                            bakePoolSizes[1].descriptorCount = 64;
                            bakePoolSizes[2].type =
                                VK_DESCRIPTOR_TYPE_SAMPLER;
                            bakePoolSizes[2].descriptorCount = 64;

                            VkDescriptorPoolCreateInfo bakePoolInfo{};
                            bakePoolInfo.sType =
                                VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                            bakePoolInfo.maxSets = 64;
                            bakePoolInfo.poolSizeCount =
                                static_cast<uint32_t>(
                                    std::size(bakePoolSizes));
                            bakePoolInfo.pPoolSizes = bakePoolSizes;
                            throwIfFailed(
                                vkCreateDescriptorPool(
                                    m_device.device(),
                                    &bakePoolInfo,
                                    nullptr,
                                    &m_environmentResources.bakeDescriptorPool),
                                "Failed to create Vulkan IBL bake descriptor pool.");
                        }

                        auto createView =
                            [&](const VulkanTexture& texture,
                                VkImageViewType type,
                                uint32_t baseMip,
                                uint32_t levelCount,
                                uint32_t layerCount)
                            {
                                VkImageViewCreateInfo viewInfo{};
                                viewInfo.sType =
                                    VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                                viewInfo.image = texture.image;
                                viewInfo.viewType = type;
                                viewInfo.format = texture.format;
                                viewInfo.subresourceRange.aspectMask =
                                    VK_IMAGE_ASPECT_COLOR_BIT;
                                viewInfo.subresourceRange.baseMipLevel =
                                    baseMip;
                                viewInfo.subresourceRange.levelCount =
                                    levelCount;
                                viewInfo.subresourceRange.layerCount =
                                    layerCount;

                                VkImageView view = VK_NULL_HANDLE;
                                throwIfFailed(
                                    vkCreateImageView(
                                        m_device.device(),
                                        &viewInfo,
                                        nullptr,
                                        &view),
                                    "Failed to create Vulkan IBL image view.");
                                return view;
                            };

                        auto transitionTexture =
                            [&](const VulkanTexture& texture,
                                VkImageLayout oldLayout,
                                VkImageLayout newLayout,
                                VkAccessFlags srcAccess,
                                VkAccessFlags dstAccess,
                                VkPipelineStageFlags srcStage,
                                VkPipelineStageFlags dstStage)
                            {
                                VkImageMemoryBarrier barrier{};
                                barrier.sType =
                                    VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                                barrier.oldLayout = oldLayout;
                                barrier.newLayout = newLayout;
                                barrier.srcAccessMask = srcAccess;
                                barrier.dstAccessMask = dstAccess;
                                barrier.srcQueueFamilyIndex =
                                    VK_QUEUE_FAMILY_IGNORED;
                                barrier.dstQueueFamilyIndex =
                                    VK_QUEUE_FAMILY_IGNORED;
                                barrier.image = texture.image;
                                barrier.subresourceRange.aspectMask =
                                    VK_IMAGE_ASPECT_COLOR_BIT;
                                barrier.subresourceRange.levelCount =
                                    texture.mipLevels;
                                barrier.subresourceRange.layerCount =
                                    texture.arrayLayers;
                                vkCmdPipelineBarrier(
                                    cmd,
                                    srcStage,
                                    dstStage,
                                    0,
                                    0,
                                    nullptr,
                                    0,
                                    nullptr,
                                    1,
                                    &barrier);
                            };

                        auto allocateSet =
                            [&](VkDescriptorSetLayout layout)
                            {
                                VkDescriptorSet set = VK_NULL_HANDLE;
                                VkDescriptorSetAllocateInfo info{};
                                info.sType =
                                    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                                info.descriptorPool =
                                    m_environmentResources.bakeDescriptorPool;
                                info.descriptorSetCount = 1;
                                info.pSetLayouts = &layout;
                                throwIfFailed(
                                    vkAllocateDescriptorSets(
                                        m_device.device(),
                                        &info,
                                        &set),
                                    "Failed to allocate Vulkan IBL descriptor set.");
                                return set;
                            };

                        auto updateSet =
                            [&](VkDescriptorSet set,
                                VkImageView source,
                                VkImageView output)
                            {
                                VkDescriptorImageInfo sourceInfo{};
                                sourceInfo.imageView = source;
                                sourceInfo.imageLayout =
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                                VkDescriptorImageInfo outputInfo{};
                                outputInfo.imageView = output;
                                outputInfo.imageLayout =
                                    VK_IMAGE_LAYOUT_GENERAL;

                                VkDescriptorImageInfo samplerInfo{};
                                samplerInfo.sampler =
                                    m_environmentResources.sampler;

                                VkWriteDescriptorSet writes[3]{};
                                writes[0].sType =
                                    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                                writes[0].dstSet = set;
                                writes[0].dstBinding = 0;
                                writes[0].descriptorCount = 1;
                                writes[0].descriptorType =
                                    VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                                writes[0].pImageInfo = &sourceInfo;

                                writes[1].sType =
                                    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                                writes[1].dstSet = set;
                                writes[1].dstBinding = 1;
                                writes[1].descriptorCount = 1;
                                writes[1].descriptorType =
                                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                                writes[1].pImageInfo = &outputInfo;

                                writes[2].sType =
                                    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                                writes[2].dstSet = set;
                                writes[2].dstBinding = 2;
                                writes[2].descriptorCount = 1;
                                writes[2].descriptorType =
                                    VK_DESCRIPTOR_TYPE_SAMPLER;
                                writes[2].pImageInfo = &samplerInfo;

                                vkUpdateDescriptorSets(
                                    m_device.device(),
                                    static_cast<uint32_t>(std::size(writes)),
                                    writes,
                                    0,
                                    nullptr);
                            };

                        auto dispatch =
                            [&](VulkanComputePipeline& pipeline,
                                VkDescriptorSet set,
                                uint32_t groupsX,
                                uint32_t groupsY,
                                uint32_t groupsZ)
                            {
                                vkCmdBindPipeline(
                                    cmd,
                                    VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline.pipeline);
                                vkCmdBindDescriptorSets(
                                    cmd,
                                    VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline.pipelineLayout,
                                    0,
                                    1,
                                    &set,
                                    0,
                                    nullptr);
                                vkCmdDispatch(
                                    cmd,
                                    groupsX,
                                    groupsY,
                                    groupsZ);
                            };

                        if (!m_environmentResources.irradiance)
                        {
                            m_environmentResources.irradiance =
                                m_resourceAllocator.createTexture({
                                    .width = m_environmentResources.irradianceSize,
                                    .height = m_environmentResources.irradianceSize,
                                    .depth = 1,
                                    .mipLevels = 1,
                                    .arrayLayers = 6,
                                    .cubeCompatible = true,
                                    .format = request.desc.format,
                                    .usage =
                                        TextureUsageFlags::Sampled |
                                        TextureUsageFlags::Storage,
                                    .memoryUsage = ResourceMemoryUsage::GpuOnly,
                                    .debugName = "Vulkan IBL irradiance cubemap"
                                    });
                            m_environmentResources.irradianceLayout =
                                VK_IMAGE_LAYOUT_UNDEFINED;
                            m_environmentResources.irradianceView =
                                createView(
                                    m_environmentResources.irradiance,
                                    VK_IMAGE_VIEW_TYPE_CUBE,
                                    0,
                                    1,
                                    6);
                            m_environmentResources.irradianceStorageView =
                                createView(
                                    m_environmentResources.irradiance,
                                    VK_IMAGE_VIEW_TYPE_2D_ARRAY,
                                    0,
                                    1,
                                    6);
                            m_environmentResources.irradianceDescriptorSet =
                                allocateSet(
                                    irradiancePipeline->descriptorSetLayout);
                            updateSet(
                                m_environmentResources.irradianceDescriptorSet,
                                m_environmentResources.cubemapView,
                                m_environmentResources.irradianceStorageView);
                        }

                        if (!m_environmentResources.prefiltered)
                        {
                            m_environmentResources.prefiltered =
                                m_resourceAllocator.createTexture({
                                    .width = m_environmentResources.prefilterSize,
                                    .height = m_environmentResources.prefilterSize,
                                    .depth = 1,
                                    .mipLevels =
                                        m_environmentResources.prefilterMipCount,
                                    .arrayLayers = 6,
                                    .cubeCompatible = true,
                                    .format = request.desc.format,
                                    .usage =
                                        TextureUsageFlags::Sampled |
                                        TextureUsageFlags::Storage,
                                    .memoryUsage = ResourceMemoryUsage::GpuOnly,
                                    .debugName = "Vulkan IBL prefiltered cubemap"
                                    });
                            m_environmentResources.prefilteredLayout =
                                VK_IMAGE_LAYOUT_UNDEFINED;
                            m_environmentResources.prefilteredView =
                                createView(
                                    m_environmentResources.prefiltered,
                                    VK_IMAGE_VIEW_TYPE_CUBE,
                                    0,
                                    m_environmentResources.prefilterMipCount,
                                    6);
                            m_environmentResources.prefilteredStorageViews.clear();
                            m_environmentResources.prefilterDescriptorSets.clear();
                            for (uint32_t mip = 0;
                                mip < m_environmentResources.prefilterMipCount;
                                ++mip)
                            {
                                VkImageView view =
                                    createView(
                                        m_environmentResources.prefiltered,
                                        VK_IMAGE_VIEW_TYPE_2D_ARRAY,
                                        mip,
                                        1,
                                        6);
                                m_environmentResources.prefilteredStorageViews
                                    .push_back(view);
                                VkDescriptorSet set =
                                    allocateSet(
                                        prefilterPipeline->descriptorSetLayout);
                                updateSet(
                                    set,
                                    m_environmentResources.cubemapView,
                                    view);
                                m_environmentResources.prefilterDescriptorSets
                                    .push_back(set);
                            }
                        }

                        if (!m_environmentResources.brdfLut)
                        {
                            m_environmentResources.brdfLut =
                                m_resourceAllocator.createTexture({
                                    .width = m_environmentResources.brdfLutSize,
                                    .height = m_environmentResources.brdfLutSize,
                                    .depth = 1,
                                    .mipLevels = 1,
                                    .arrayLayers = 1,
                                    .format = request.desc.format,
                                    .usage =
                                        TextureUsageFlags::Sampled |
                                        TextureUsageFlags::Storage,
                                    .memoryUsage = ResourceMemoryUsage::GpuOnly,
                                    .debugName = "Vulkan IBL BRDF LUT"
                                    });
                            m_environmentResources.brdfLutLayout =
                                VK_IMAGE_LAYOUT_UNDEFINED;
                            m_environmentResources.brdfLutView =
                                createView(
                                    m_environmentResources.brdfLut,
                                    VK_IMAGE_VIEW_TYPE_2D,
                                    0,
                                    1,
                                    1);
                            m_environmentResources.brdfLutStorageView =
                                createView(
                                    m_environmentResources.brdfLut,
                                    VK_IMAGE_VIEW_TYPE_2D,
                                    0,
                                    1,
                                    1);
                            m_environmentResources.brdfLutDescriptorSet =
                                allocateSet(brdfPipeline->descriptorSetLayout);
                            updateSet(
                                m_environmentResources.brdfLutDescriptorSet,
                                m_environmentResources.cubemapView,
                                m_environmentResources.brdfLutStorageView);
                        }

                        transitionTexture(
                            m_environmentResources.irradiance,
                            m_environmentResources.irradianceLayout,
                            VK_IMAGE_LAYOUT_GENERAL,
                            0,
                            VK_ACCESS_SHADER_WRITE_BIT,
                            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                        m_environmentResources.irradianceLayout =
                            VK_IMAGE_LAYOUT_GENERAL;
                        dispatch(
                            *irradiancePipeline,
                            m_environmentResources.irradianceDescriptorSet,
                            (m_environmentResources.irradianceSize + 7u) / 8u,
                            (m_environmentResources.irradianceSize + 7u) / 8u,
                            6u);
                        transitionTexture(
                            m_environmentResources.irradiance,
                            VK_IMAGE_LAYOUT_GENERAL,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_ACCESS_SHADER_WRITE_BIT,
                            VK_ACCESS_SHADER_READ_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
                        m_environmentResources.irradianceLayout =
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                        transitionTexture(
                            m_environmentResources.prefiltered,
                            m_environmentResources.prefilteredLayout,
                            VK_IMAGE_LAYOUT_GENERAL,
                            0,
                            VK_ACCESS_SHADER_WRITE_BIT,
                            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                        m_environmentResources.prefilteredLayout =
                            VK_IMAGE_LAYOUT_GENERAL;
                        for (uint32_t mip = 0;
                            mip < m_environmentResources.prefilterMipCount;
                            ++mip)
                        {
                            const uint32_t mipSize =
                                std::max(
                                    m_environmentResources.prefilterSize >> mip,
                                    1u);
                            dispatch(
                                *prefilterPipeline,
                                m_environmentResources
                                .prefilterDescriptorSets[mip],
                                (mipSize + 7u) / 8u,
                                (mipSize + 7u) / 8u,
                                6u);
                        }
                        transitionTexture(
                            m_environmentResources.prefiltered,
                            VK_IMAGE_LAYOUT_GENERAL,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_ACCESS_SHADER_WRITE_BIT,
                            VK_ACCESS_SHADER_READ_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
                        m_environmentResources.prefilteredLayout =
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                        transitionTexture(
                            m_environmentResources.brdfLut,
                            m_environmentResources.brdfLutLayout,
                            VK_IMAGE_LAYOUT_GENERAL,
                            0,
                            VK_ACCESS_SHADER_WRITE_BIT,
                            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                        m_environmentResources.brdfLutLayout =
                            VK_IMAGE_LAYOUT_GENERAL;
                        dispatch(
                            *brdfPipeline,
                            m_environmentResources.brdfLutDescriptorSet,
                            (m_environmentResources.brdfLutSize + 7u) / 8u,
                            (m_environmentResources.brdfLutSize + 7u) / 8u,
                            1u);
                        transitionTexture(
                            m_environmentResources.brdfLut,
                            VK_IMAGE_LAYOUT_GENERAL,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_ACCESS_SHADER_WRITE_BIT,
                            VK_ACCESS_SHADER_READ_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
                        m_environmentResources.brdfLutLayout =
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                        m_environmentResources.iblBaked = true;

                    });

                if (!m_environmentResources.converted)
                {
                    result.error = "SourceNotReady";
                    results.push_back(std::move(result));
                    continue;
                }

                result.success = true;
                result.resources.environmentCubemap =
                    TextureHandle{ request.handle.index };
                result.resources.irradianceCubemap =
                    TextureHandle{ request.handle.index + 1u };
                result.resources.prefilteredCubemap =
                    TextureHandle{ request.handle.index + 2u };
                result.resources.brdfLut =
                    TextureHandle{ request.handle.index + 3u };
                result.resources.environmentBindlessIndex = UINT32_MAX;
                result.resources.irradianceBindlessIndex = UINT32_MAX;
                result.resources.prefilteredBindlessIndex = UINT32_MAX;
                result.resources.brdfLutBindlessIndex = UINT32_MAX;
                result.resources.prefilteredMipCount =
                    m_environmentResources.prefilterMipCount;

                spdlog::info(
                    "[Vulkan] IBL bake request={} produced env={} irradiance={} prefilter={} mips={} brdf={}",
                    request.requestId,
                    m_environmentResources.cubemapSize,
                    m_environmentResources.irradianceSize,
                    m_environmentResources.prefilterSize,
                    m_environmentResources.prefilterMipCount,
                    m_environmentResources.brdfLutSize);
            }
            catch (const std::exception& e)
            {
                result.success = false;
                result.error = e.what();
            }

            results.push_back(std::move(result));
        }

        return results;
    }

    uint32_t VulkanBackend::requestTexture(
        AssetHandle modelHandle,
        uint32_t imageIndex,
        const ImageAsset& image,
        TextureTransferFunction transfer,
        bool asynchronous)
    {
        const uint64_t key = uploadedTextureKey(modelHandle, imageIndex, transfer);
        if (auto it = m_uploadedTextures.find(key);
            it != m_uploadedTextures.end())
        {
            return it->second.descriptorIndex;
        }

        ImageAsset uploadImage = image;
        if (transfer != TextureTransferFunction::Unknown &&
            uploadImage.format == ImageFormat::RGBA8)
        {
            uploadImage.srgb = transfer == TextureTransferFunction::SRGB;
        }

        const uint64_t byteSize = imageByteSize(uploadImage);
        VulkanBuffer staging =
            m_resourceAllocator.createBuffer({
                .size = byteSize,
                .usage = BufferUsageFlags::TransferSrc,
                .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                .mappedAtCreation = true,
                .debugName = "Vulkan texture upload staging"
            });

        std::memcpy(
            staging.mapped,
            uploadImage.pixels.data(),
            static_cast<size_t>(byteSize));
        m_resourceAllocator.flush(staging, 0, byteSize);

        UploadedTexture uploaded{};
        uploaded.descriptorIndex =
            static_cast<uint32_t>(m_uploadedTextures.size());
        if (uploaded.descriptorIndex >= MaxBindlessTextures)
        {
            throw std::runtime_error("Vulkan bindless texture array exhausted.");
        }
        uploaded.texture =
            m_resourceAllocator.createTexture(
                uploadImage,
                TextureUsageFlags::Sampled | TextureUsageFlags::TransferDst,
                "Vulkan bindless model texture");

        auto recordUpload = [&](VkCommandBuffer cmd)
            {
                transitionImage(
                    cmd,
                    uploaded.texture.image,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    0,
                    VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT);

                VkBufferImageCopy copy{};
                copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                copy.imageSubresource.mipLevel = 0;
                copy.imageSubresource.baseArrayLayer = 0;
                copy.imageSubresource.layerCount = 1;
                copy.imageExtent = uploaded.texture.extent;

                vkCmdCopyBufferToImage(
                    cmd,
                    staging.buffer,
                    uploaded.texture.image,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1,
                    &copy);

                transitionImage(
                    cmd,
                    uploaded.texture.image,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_ACCESS_TRANSFER_WRITE_BIT,
                    asynchronous ? 0 : VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    asynchronous
                        ? VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT
                        : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            };
        if (asynchronous)
        {
            m_uploadScheduler.record(recordUpload);
        }
        else
        {
            m_commandSystem.immediateSubmit(
                m_device.graphicsQueue(), std::move(recordUpload));
        }

        uploaded.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = uploaded.texture.image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = uploaded.texture.format;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.layerCount = 1;

        throwIfFailed(
            vkCreateImageView(
                m_device.device(),
                &viewInfo,
                nullptr,
                &uploaded.view),
            "Failed to create Vulkan bindless texture view.");

        if (asynchronous)
        {
            m_retirementQueue.retire(std::move(staging));
        }
        else
        {
            m_resourceAllocator.destroyBuffer(staging);
        }

        m_uploadedTextures.emplace(key, std::move(uploaded));
        return static_cast<uint32_t>(m_uploadedTextures.size() - 1u);
    }

    uint32_t VulkanBackend::requestSampler(const SamplerAsset* sampler)
    {
        const uint64_t key = samplerKey(sampler);
        if (auto it = m_uploadedSamplers.find(key);
            it != m_uploadedSamplers.end())
        {
            return it->second.descriptorIndex;
        }

        auto toFilter = [](TextureFilterMode mode)
        {
            return mode == TextureFilterMode::Nearest ||
                mode == TextureFilterMode::NearestMipmapNearest ||
                mode == TextureFilterMode::NearestMipmapLinear
                    ? VK_FILTER_NEAREST
                    : VK_FILTER_LINEAR;
        };

        auto toAddress = [](TextureWrapMode mode)
        {
            switch (mode)
            {
            case TextureWrapMode::MirroredRepeat:
                return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
            case TextureWrapMode::ClampToEdge:
                return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            case TextureWrapMode::Repeat:
                break;
            }
            return VK_SAMPLER_ADDRESS_MODE_REPEAT;
        };

        UploadedSampler uploaded{};
        uploaded.descriptorIndex =
            static_cast<uint32_t>(m_uploadedSamplers.size());
        if (uploaded.descriptorIndex >= MaxBindlessSamplers)
        {
            throw std::runtime_error("Vulkan bindless sampler array exhausted.");
        }
        VkSamplerCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        info.magFilter = sampler ? toFilter(sampler->magFilter) : VK_FILTER_LINEAR;
        info.minFilter = sampler ? toFilter(sampler->minFilter) : VK_FILTER_LINEAR;
        info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        info.addressModeU = sampler ? toAddress(sampler->wrapU) : VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.addressModeV = sampler ? toAddress(sampler->wrapV) : VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.maxLod = VK_LOD_CLAMP_NONE;

        throwIfFailed(
            vkCreateSampler(
                m_device.device(),
                &info,
                nullptr,
                &uploaded.sampler),
            "Failed to create Vulkan bindless sampler.");

        const uint32_t descriptorIndex = uploaded.descriptorIndex;
        m_uploadedSamplers.emplace(key, uploaded);
        return descriptorIndex;
    }

    VulkanUploadedModel* VulkanBackend::requestModel(
        AssetHandle handle,
        const AssetManager& assets)
    {
        if (!handle)
        {
            return nullptr;
        }

        auto [it, inserted] =
            m_uploadedModels.try_emplace(handle);
        VulkanUploadedModel& uploaded = it->second;
        if (uploaded.uploaded)
        {
            return &uploaded;
        }

        const ModelAsset* model = assets.model(handle);
        if (!model ||
            model->vertices.empty() ||
            model->indices.empty())
        {
            return nullptr;
        }

        const VkDeviceSize vertexBytes =
            static_cast<VkDeviceSize>(model->vertices.size()) *
            sizeof(AssetVertex);
        const VkDeviceSize indexBytes =
            static_cast<VkDeviceSize>(model->indices.size()) *
            sizeof(uint32_t);

        VulkanBuffer vertexStaging =
            m_resourceAllocator.createBuffer({
                .size = vertexBytes,
                .usage = BufferUsageFlags::TransferSrc,
                .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                .mappedAtCreation = true,
                .debugName = "Vulkan model vertex staging"
            });

        VulkanBuffer indexStaging =
            m_resourceAllocator.createBuffer({
                .size = indexBytes,
                .usage = BufferUsageFlags::TransferSrc,
                .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                .mappedAtCreation = true,
                .debugName = "Vulkan model index staging"
            });

        std::memcpy(
            vertexStaging.mapped,
            model->vertices.data(),
            static_cast<size_t>(vertexBytes));
        std::memcpy(
            indexStaging.mapped,
            model->indices.data(),
            static_cast<size_t>(indexBytes));

        m_resourceAllocator.flush(
            vertexStaging,
            0,
            vertexBytes);
        m_resourceAllocator.flush(
            indexStaging,
            0,
            indexBytes);

        uploaded.vertexBuffer =
            m_resourceAllocator.createBuffer({
                .size = vertexBytes,
                .usage =
                    BufferUsageFlags::Vertex |
                    BufferUsageFlags::TransferDst,
                .memoryUsage = ResourceMemoryUsage::GpuOnly,
                .mappedAtCreation = false,
                .debugName = "Vulkan model vertex buffer"
            });

        uploaded.indexBuffer =
            m_resourceAllocator.createBuffer({
                .size = indexBytes,
                .usage =
                    BufferUsageFlags::Index |
                    BufferUsageFlags::TransferDst,
                .memoryUsage = ResourceMemoryUsage::GpuOnly,
                .mappedAtCreation = false,
                .debugName = "Vulkan model index buffer"
            });

        m_uploadScheduler.record(
            [&](VkCommandBuffer cmd)
            {
                VkBufferCopy vertexCopy{};
                vertexCopy.size = vertexBytes;
                vkCmdCopyBuffer(
                    cmd,
                    vertexStaging.buffer,
                    uploaded.vertexBuffer.buffer,
                    1,
                    &vertexCopy);

                VkBufferCopy indexCopy{};
                indexCopy.size = indexBytes;
                vkCmdCopyBuffer(
                    cmd,
                    indexStaging.buffer,
                    uploaded.indexBuffer.buffer,
                    1,
                    &indexCopy);

                VkBufferMemoryBarrier barriers[2]{};
                barriers[0].sType =
                    VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barriers[0].srcAccessMask =
                    VK_ACCESS_TRANSFER_WRITE_BIT;
                barriers[0].dstAccessMask = 0;
                barriers[0].srcQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED;
                barriers[0].dstQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED;
                barriers[0].buffer = uploaded.vertexBuffer.buffer;
                barriers[0].offset = 0;
                barriers[0].size = vertexBytes;

                barriers[1].sType =
                    VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barriers[1].srcAccessMask =
                    VK_ACCESS_TRANSFER_WRITE_BIT;
                barriers[1].dstAccessMask = 0;
                barriers[1].srcQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED;
                barriers[1].dstQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED;
                barriers[1].buffer = uploaded.indexBuffer.buffer;
                barriers[1].offset = 0;
                barriers[1].size = indexBytes;

                vkCmdPipelineBarrier(
                    cmd,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                    0,
                    0,
                    nullptr,
                    2,
                    barriers,
                    0,
                    nullptr);
            });

        m_retirementQueue.retire(std::move(vertexStaging));
        m_retirementQueue.retire(std::move(indexStaging));

        uploaded.samplerDescriptorIndices.assign(
            model->samplers.size(),
            UINT32_MAX);
        for (size_t samplerIndex = 0;
             samplerIndex < model->samplers.size();
             ++samplerIndex)
        {
            uploaded.samplerDescriptorIndices[samplerIndex] =
                requestSampler(&model->samplers[samplerIndex]);
        }
        const uint32_t defaultSampler = requestSampler(nullptr);

        auto textureDescriptor = [&](const MaterialTextureSlot& slot)
        {
            if (slot.textureIndex < 0 ||
                static_cast<size_t>(slot.textureIndex) >= model->textures.size())
            {
                return UINT32_MAX;
            }

            const TextureAsset& texture =
                model->textures[static_cast<size_t>(slot.textureIndex)];
            if (texture.imageIndex < 0 ||
                static_cast<size_t>(texture.imageIndex) >= model->images.size())
            {
                return UINT32_MAX;
            }

            return requestTexture(
                handle,
                static_cast<uint32_t>(texture.imageIndex),
                model->images[static_cast<size_t>(texture.imageIndex)],
                slot.transfer);
        };

        auto samplerDescriptor = [&](const MaterialTextureSlot& slot)
        {
            if (slot.textureIndex < 0 ||
                static_cast<size_t>(slot.textureIndex) >= model->textures.size())
            {
                return defaultSampler;
            }

            const TextureAsset& texture =
                model->textures[static_cast<size_t>(slot.textureIndex)];
            if (texture.samplerIndex < 0 ||
                static_cast<size_t>(texture.samplerIndex) >=
                    uploaded.samplerDescriptorIndices.size())
            {
                return defaultSampler;
            }

            const uint32_t descriptor =
                uploaded.samplerDescriptorIndices[
                    static_cast<size_t>(texture.samplerIndex)];
            return descriptor != UINT32_MAX ? descriptor : defaultSampler;
        };

        uploaded.materials.reserve(
            std::max<size_t>(1, model->materials.size()));
        for (const MaterialAsset& src : model->materials)
        {
            GpuMaterialData dst = makeGpuMaterialData(src);
            dst.baseColorTextureIndex = textureDescriptor(src.baseColorTexture);
            dst.normalTextureIndex = textureDescriptor(src.normalTexture);
            dst.metallicRoughnessTextureIndex =
                textureDescriptor(src.metallicRoughnessTexture);
            dst.occlusionTextureIndex =
                textureDescriptor(src.occlusionTexture);
            dst.emissiveTextureIndex =
                textureDescriptor(src.emissiveTexture);
            dst.baseColorSamplerIndex =
                samplerDescriptor(src.baseColorTexture);
            dst.normalSamplerIndex =
                samplerDescriptor(src.normalTexture);
            dst.metallicRoughnessSamplerIndex =
                samplerDescriptor(src.metallicRoughnessTexture);
            dst.occlusionSamplerIndex =
                samplerDescriptor(src.occlusionTexture);
            dst.emissiveSamplerIndex =
                samplerDescriptor(src.emissiveTexture);
            uploaded.materials.push_back(dst);
        }
        if (uploaded.materials.empty())
        {
            uploaded.materials.push_back(GpuMaterialData{});
        }

        std::vector<glm::mat4> meshNodeTransforms(
            model->meshes.size(),
            glm::mat4(1.0f));

        for (const NodeAsset& node : model->nodes)
        {
            if (node.meshIndex >= 0 &&
                static_cast<size_t>(node.meshIndex) <
                    meshNodeTransforms.size())
            {
                meshNodeTransforms[static_cast<size_t>(node.meshIndex)] =
                    node.worldMatrix;
            }
        }

        for (size_t meshIndex = 0;
             meshIndex < model->meshes.size();
             ++meshIndex)
        {
            const MeshAsset& mesh = model->meshes[meshIndex];
            for (const MeshPrimitiveAsset& primitive : mesh.primitives)
            {
                GpuMesh gpuMesh{};
                gpuMesh.materialIndex =
                    primitive.materialIndex != UINT32_MAX
                        ? primitive.materialIndex
                        : 0u;
                gpuMesh.indexCount = primitive.indexCount;
                gpuMesh.firstIndex = primitive.firstIndex;
                gpuMesh.vertexOffset = primitive.firstVertex;
                const glm::vec3 center =
                    (primitive.bounds.min + primitive.bounds.max) * 0.5f;
                const glm::vec3 extent =
                    (primitive.bounds.max - primitive.bounds.min) * 0.5f;
                gpuMesh.bounds.centerRadius =
                    glm::vec4(center, glm::length(extent));
                uploaded.meshes.push_back(gpuMesh);
                uploaded.meshTransforms.push_back(
                    meshNodeTransforms[meshIndex]);
            }
        }

        uploaded.uploaded = true;
        return &uploaded;
    }

    bool VulkanBackend::prepareSceneResources(
        const FrameContext& ctx,
        const SceneRenderView& scene,
        GraphicsPipelineHandle pipelineHandle,
        bool updateGraphicsDescriptors)
    {
        if (!ctx.services || !ctx.services->assetManager)
        {
            return false;
        }
        AssetManager& assets = *ctx.services->assetManager;

        if (m_gpuScene.frameSlotCount() == 0)
        {
            return false;
        }
        const uint32_t frameSlot = static_cast<uint32_t>(
            ctx.frameIndex % m_gpuScene.frameSlotCount());

        m_uploadedModels.reserve(
            m_uploadedModels.size() + scene.models.size());

        const VulkanGpuScene::PrepareResult result = m_gpuScene.prepare(
            ctx.frameIndex,
            scene,
            frameSlot,
            [this, &assets](AssetHandle handle) -> GpuSceneModelView
            {
                VulkanUploadedModel* model = requestModel(handle, assets);
                if (!model)
                {
                    return {};
                }
                return GpuSceneModelView{
                    handle, model->meshes, model->meshTransforms,
                    model->materials };
            },
            [this, &ctx, &scene](
                uint32_t visibleLightCount,
                uint32_t instanceBoundsCount,
                uint32_t geometryBinCount) -> GpuFrameData
            {
                // Reads cluster grid dimensions this call establishes, so it
                // must run before the fields below are consumed.
                ensureClusteredForwardResources();

                GpuFrameData frameData{};
                frameData.view = scene.camera.view;
                frameData.projection = glm::perspectiveRH_ZO(
                    scene.camera.verticalFovRadians,
                    scene.camera.aspectRatio,
                    scene.camera.nearPlane,
                    scene.camera.farPlane);
                frameData.projection[1][1] *= -1.0f;
                frameData.viewProjection = frameData.projection * frameData.view;
                frameData.cameraPosition = scene.camera.position;
                frameData.time = ctx.timeSinceStart;
                frameData.environmentEnabled =
                    m_environmentResources.iblBaked ? 1u : 0u;
                frameData.prefilteredMipCount =
                    m_environmentResources.prefilterMipCount;
                frameData.environmentIntensity =
                    scene.environment.settings.intensity;
                frameData.environmentExposure =
                    scene.environment.settings.skyboxExposure;

                for (const SceneLightRenderItem& light : scene.lights)
                {
                    if (light.type == LightType::Directional)
                    {
                        frameData.lightDirection = light.direction;
                        frameData.lightColor = light.color;
                        frameData.lightIntensity = light.intensity;
                        break;
                    }
                }
                for (const SceneLightRenderItem& light : scene.lights)
                {
                    if (light.type != LightType::Point ||
                        frameData.pointLightCount >= MaxGpuPointLights)
                    {
                        continue;
                    }

                    const uint32_t lightIndex = frameData.pointLightCount++;
                    frameData.pointLightPositionRange[lightIndex] =
                        glm::vec4(light.position, light.range);
                    frameData.pointLightColorIntensity[lightIndex] =
                        glm::vec4(light.color, light.intensity);
                }
                frameData.pointLightCount =
                    std::min<uint32_t>(
                        visibleLightCount, ClusteredForwardMaxVisibleLights);
                frameData.clusterDimensions = glm::uvec4(
                    m_clusteredForwardResources.clusterCountX,
                    m_clusteredForwardResources.clusterCountY,
                    m_clusteredForwardResources.clusterCountZ,
                    m_clusteredForwardResources.clusterCount);
                frameData.clusterConfig = glm::uvec4(
                    ClusteredForwardTileSizeX,
                    ClusteredForwardTileSizeY,
                    ClusteredForwardMaxLightsPerCluster,
                    m_clusteredForwardHeatmapEnabled ? 1u : 0u);
                frameData.renderExtentAndHiZ = glm::uvec4(
                    m_clusteredForwardResources.width,
                    m_clusteredForwardResources.height,
                    m_clusteredForwardResources.hiZMipCount,
                    0u);
                frameData.cullingConfig = glm::uvec4(
                    std::min<uint32_t>(
                        instanceBoundsCount, ClusteredForwardMaxGpuCullInstances),
                    1u,
                    geometryBinCount,
                    0u);
                frameData.cameraNearFar = glm::vec4(
                    scene.camera.nearPlane, scene.camera.farPlane, 0.0f, 0.0f);
                return frameData;
            });

        if (!result.hasData)
        {
            return false;
        }

        VulkanGpuSceneFrameResources& frameResources =
            m_gpuScene.frameResources(frameSlot);
        VulkanGraphicsPipeline* pipeline = updateGraphicsDescriptors
            ? m_pipelineManager.graphicsPipeline(pipelineHandle)
            : nullptr;
        if (updateGraphicsDescriptors &&
            (result.descriptorsDirty ||
            frameResources.descriptorSet == VK_NULL_HANDLE ||
            (pipeline &&
                frameResources.descriptorLayout !=
                pipeline->desc.bindingLayout) ||
            frameResources.bindlessTextureCount != m_uploadedTextures.size() ||
            frameResources.bindlessSamplerCount != m_uploadedSamplers.size() ||
            frameResources.environmentVersion != scene.environment.version ||
            frameResources.iblBaked != m_environmentResources.iblBaked))
        {
            updateFrameDescriptors(frameResources, pipelineHandle);
            frameResources.environmentVersion = scene.environment.version;
            frameResources.iblBaked = m_environmentResources.iblBaked;
        }

        return true;
    }

    void VulkanBackend::updateFrameDescriptors(
        VulkanGpuSceneFrameResources& resources,
        GraphicsPipelineHandle pipelineHandle)
    {
        VulkanGraphicsPipeline* pipeline =
            m_pipelineManager.graphicsPipeline(pipelineHandle);
        if (!pipeline)
        {
            return;
        }

        if (resources.descriptorPool != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(
                m_device.device(),
                resources.descriptorPool,
                nullptr);
            resources.descriptorPool = VK_NULL_HANDLE;
            resources.descriptorSet = VK_NULL_HANDLE;
        }

        const bool clusteredLayout =
            pipeline->desc.bindingLayout ==
            PipelineBindingLayoutKind::ClusteredForward;

        VkDescriptorPoolSize poolSizes[4]{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = 1;
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[1].descriptorCount = clusteredLayout ? 8u : 3u;
        poolSizes[2].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        poolSizes[2].descriptorCount = MaxBindlessTextures + 3u;
        poolSizes[3].type = VK_DESCRIPTOR_TYPE_SAMPLER;
        poolSizes[3].descriptorCount = MaxBindlessSamplers + 1u;

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount =
            static_cast<uint32_t>(std::size(poolSizes));
        poolInfo.pPoolSizes = poolSizes;

        if (vkCreateDescriptorPool(
                m_device.device(),
                &poolInfo,
                nullptr,
                &resources.descriptorPool) != VK_SUCCESS)
        {
            throw std::runtime_error(
                "Failed to create Vulkan forward descriptor pool.");
        }

        VkDescriptorSetAllocateInfo allocateInfo{};
        allocateInfo.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocateInfo.descriptorPool = resources.descriptorPool;
        allocateInfo.descriptorSetCount = 1;
        allocateInfo.pSetLayouts = &pipeline->descriptorSetLayout;

        if (vkAllocateDescriptorSets(
                m_device.device(),
                &allocateInfo,
                &resources.descriptorSet) != VK_SUCCESS)
        {
            throw std::runtime_error(
                "Failed to allocate Vulkan forward descriptor set.");
        }

        VkDescriptorBufferInfo frameInfo{};
        frameInfo.buffer = resources.frameConstants.buffer;
        frameInfo.offset = 0;
        frameInfo.range = sizeof(GpuFrameData);

        VkDescriptorBufferInfo objectInfo{};
        objectInfo.buffer = resources.objects.buffer;
        objectInfo.offset = 0;
        objectInfo.range = resources.objects.size;

        VkDescriptorBufferInfo materialInfo{};
        materialInfo.buffer = resources.materials.buffer;
        materialInfo.offset = 0;
        materialInfo.range = resources.materials.size;

        std::vector<VkDescriptorImageInfo> textureInfos(
            m_uploadedTextures.size());
        for (const auto& [key, texture] : m_uploadedTextures)
        {
            if (texture.descriptorIndex >= textureInfos.size())
            {
                continue;
            }

            textureInfos[texture.descriptorIndex].imageView = texture.view;
            textureInfos[texture.descriptorIndex].imageLayout = texture.layout;
        }

        std::vector<VkDescriptorImageInfo> samplerInfos(
            m_uploadedSamplers.size());
        for (const auto& [key, sampler] : m_uploadedSamplers)
        {
            if (sampler.descriptorIndex >= samplerInfos.size())
            {
                continue;
            }

            samplerInfos[sampler.descriptorIndex].sampler = sampler.sampler;
        }

        std::vector<VkWriteDescriptorSet> writes;
        writes.reserve(clusteredLayout ? 10u : 5u);
        writes.resize(3);
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = resources.descriptorSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].pBufferInfo = &frameInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = resources.descriptorSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &objectInfo;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = resources.descriptorSet;
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo = &materialInfo;

        VkDescriptorBufferInfo drawMetadataInfo{};
        drawMetadataInfo.buffer = m_gpuDrivenBuffers.drawMetadata;
        drawMetadataInfo.range = m_gpuDrivenBuffers.drawMetadataSize;
        VkWriteDescriptorSet drawMetadataWrite{};
        drawMetadataWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        drawMetadataWrite.dstSet = resources.descriptorSet;
        drawMetadataWrite.dstBinding = 25;
        drawMetadataWrite.descriptorCount = 1;
        drawMetadataWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        drawMetadataWrite.pBufferInfo = &drawMetadataInfo;
        writes.push_back(drawMetadataWrite);

        VkDescriptorBufferInfo clusterBoundsInfo{};
        VkDescriptorBufferInfo clusterGridInfo{};
        VkDescriptorBufferInfo clusterIndicesInfo{};
        VkDescriptorBufferInfo visibleLightsInfo{};
        VkDescriptorBufferInfo clusterCounterInfo{};
        if (clusteredLayout)
        {
            clusterBoundsInfo.buffer = m_clusterBuffers.bounds;
            clusterBoundsInfo.range = m_clusterBuffers.boundsSize;
            clusterGridInfo.buffer = m_clusterBuffers.lightGrid;
            clusterGridInfo.range = m_clusterBuffers.lightGridSize;
            clusterIndicesInfo.buffer = m_clusterBuffers.lightIndices;
            clusterIndicesInfo.range = m_clusterBuffers.lightIndicesSize;
            visibleLightsInfo.buffer = m_clusterBuffers.visibleLights;
            visibleLightsInfo.range = m_clusterBuffers.visibleLightsSize;
            clusterCounterInfo.buffer = m_clusterBuffers.lightCounter;
            clusterCounterInfo.range = m_clusterBuffers.lightCounterSize;

            const VkDescriptorBufferInfo* infos[] =
            {
                &clusterBoundsInfo,
                &clusterGridInfo,
                &clusterIndicesInfo,
                &visibleLightsInfo,
                &clusterCounterInfo
            };
            for (uint32_t i = 0; i < static_cast<uint32_t>(std::size(infos)); ++i)
            {
                VkWriteDescriptorSet write{};
                write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write.dstSet = resources.descriptorSet;
                write.dstBinding = 10u + i;
                write.descriptorCount = 1;
                write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                write.pBufferInfo = infos[i];
                writes.push_back(write);
            }
        }

        if (!textureInfos.empty())
        {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = resources.descriptorSet;
            write.dstBinding = 3;
            write.descriptorCount =
                static_cast<uint32_t>(textureInfos.size());
            write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            write.pImageInfo = textureInfos.data();
            writes.push_back(write);
        }

        if (!samplerInfos.empty())
        {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = resources.descriptorSet;
            write.dstBinding = 100;
            write.descriptorCount =
                static_cast<uint32_t>(samplerInfos.size());
            write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            write.pImageInfo = samplerInfos.data();
            writes.push_back(write);
        }

        if (m_environmentResources.iblBaked &&
            m_environmentResources.irradianceView != VK_NULL_HANDLE &&
            m_environmentResources.prefilteredView != VK_NULL_HANDLE &&
            m_environmentResources.brdfLutView != VK_NULL_HANDLE &&
            m_environmentResources.sampler != VK_NULL_HANDLE)
        {
            VkDescriptorImageInfo irradianceInfo{};
            irradianceInfo.imageView = m_environmentResources.irradianceView;
            irradianceInfo.imageLayout =
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorImageInfo prefilteredInfo{};
            prefilteredInfo.imageView = m_environmentResources.prefilteredView;
            prefilteredInfo.imageLayout =
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorImageInfo brdfLutInfo{};
            brdfLutInfo.imageView = m_environmentResources.brdfLutView;
            brdfLutInfo.imageLayout =
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorImageInfo iblSamplerInfo{};
            iblSamplerInfo.sampler = m_environmentResources.sampler;

            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = resources.descriptorSet;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;

            write.dstBinding = MaxBindlessTextures + 2u;
            write.pImageInfo = &irradianceInfo;
            writes.push_back(write);

            write.dstBinding = MaxBindlessTextures + 3u;
            write.pImageInfo = &prefilteredInfo;
            writes.push_back(write);

            write.dstBinding = MaxBindlessTextures + 4u;
            write.pImageInfo = &brdfLutInfo;
            writes.push_back(write);

            VkWriteDescriptorSet samplerWrite{};
            samplerWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            samplerWrite.dstSet = resources.descriptorSet;
            samplerWrite.dstBinding = MaxBindlessSamplers;
            samplerWrite.descriptorCount = 1;
            samplerWrite.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            samplerWrite.pImageInfo = &iblSamplerInfo;
            writes.push_back(samplerWrite);
        }

        vkUpdateDescriptorSets(
            m_device.device(),
            static_cast<uint32_t>(writes.size()),
            writes.data(),
            0,
            nullptr);

        resources.bindlessTextureCount =
            static_cast<uint32_t>(m_uploadedTextures.size());
        resources.bindlessSamplerCount =
            static_cast<uint32_t>(m_uploadedSamplers.size());
        resources.environmentVersion = UINT64_MAX;
        resources.iblBaked = m_environmentResources.iblBaked;
        resources.descriptorLayout = pipeline->desc.bindingLayout;
    }

    void VulkanBackend::initImGui(Window& window)
    {
        if (m_imguiEnabled)
        {
            return;
        }

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();

        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.IniFilename = nullptr;

        ImGui::StyleColorsDark();

        auto* glfwWindow =
            static_cast<GLFWwindow*>(window.getNativeHandle());
        if (!ImGui_ImplGlfw_InitForVulkan(glfwWindow, true))
        {
            throw std::runtime_error(
                "Failed to initialize ImGui GLFW backend.");
        }

        const VkFormat colorFormat = m_swapchain.format();
        VkPipelineRenderingCreateInfoKHR renderingInfo{};
        renderingInfo.sType =
            VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachmentFormats = &colorFormat;

        const uint32_t imageCount =
            std::max(2u, static_cast<uint32_t>(m_swapchain.images().size()));

        ImGui_ImplVulkan_InitInfo initInfo{};
        initInfo.ApiVersion = VK_API_VERSION_1_3;
        initInfo.Instance = m_instance.instance();
        initInfo.PhysicalDevice = m_adapter.device();
        initInfo.Device = m_device.device();
        initInfo.QueueFamily = m_adapter.info().queueFamilies.graphics;
        initInfo.Queue = m_device.graphicsQueue();
        initInfo.DescriptorPoolSize = 64;
        initInfo.MinImageCount = imageCount;
        initInfo.ImageCount = imageCount;
        initInfo.UseDynamicRendering = true;
        initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        initInfo.PipelineInfoMain.PipelineRenderingCreateInfo = renderingInfo;

        if (!ImGui_ImplVulkan_Init(&initInfo))
        {
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
            throw std::runtime_error(
                "Failed to initialize ImGui Vulkan backend.");
        }

        m_imguiEnabled = true;
    }

    void VulkanBackend::shutdownImGui()
    {
        if (!m_imguiEnabled)
        {
            return;
        }

        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        m_imguiIniPath.clear();

        m_imguiEnabled = false;
        m_imguiFrameActive = false;
    }

    bool VulkanBackend::beginDebugGuiFrame()
    {
        if (!m_imguiEnabled || m_imguiFrameActive)
        {
            return false;
        }

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        m_imguiFrameActive = true;
        return true;
    }

    void VulkanBackend::ensureHiZDebugDescriptors(VulkanGraphResourceEntry& hiZ)
    {
        if (!m_imguiEnabled || hiZ.mipViews.empty())
        {
            return;
        }

        // Rebuild when the Hi-Z image was recreated (e.g. on resize): the
        // cached debug views/descriptors reference the retired image.
        if (!m_clusteredForwardResources.hiZDebugDescriptors.empty())
        {
            if (m_clusteredForwardResources.hiZDebugGeneration == hiZ.generation)
            {
                return;
            }
            destroyHiZDebugDescriptors();
        }
        m_clusteredForwardResources.hiZDebugGeneration = hiZ.generation;

        if (m_clusteredForwardResources.hiZDebugSampler == VK_NULL_HANDLE)
        {
            VkSamplerCreateInfo samplerInfo{};
            samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            samplerInfo.magFilter = VK_FILTER_LINEAR;
            samplerInfo.minFilter = VK_FILTER_LINEAR;
            samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInfo.maxLod = 0.0f;
            if (vkCreateSampler(
                    m_device.device(),
                    &samplerInfo,
                    nullptr,
                    &m_clusteredForwardResources.hiZDebugSampler) !=
                VK_SUCCESS)
            {
                return;
            }
        }

        m_clusteredForwardResources.hiZDebugViews.reserve(hiZ.mipViews.size());
        m_clusteredForwardResources.hiZDebugDescriptors.reserve(hiZ.mipViews.size());
        for (uint32_t mip = 0; mip < hiZ.mipLevels; ++mip)
        {
            VkImageViewCreateInfo viewInfo{};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = hiZ.texture.image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = hiZ.texture.format;
            viewInfo.components.r = VK_COMPONENT_SWIZZLE_R;
            viewInfo.components.g = VK_COMPONENT_SWIZZLE_R;
            viewInfo.components.b = VK_COMPONENT_SWIZZLE_R;
            viewInfo.components.a = VK_COMPONENT_SWIZZLE_ONE;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.baseMipLevel = mip;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.baseArrayLayer = 0;
            viewInfo.subresourceRange.layerCount = 1;

            VkImageView debugView = VK_NULL_HANDLE;
            if (vkCreateImageView(
                    m_device.device(),
                    &viewInfo,
                    nullptr,
                    &debugView) != VK_SUCCESS)
            {
                continue;
            }

            m_clusteredForwardResources.hiZDebugViews.push_back(debugView);
            m_clusteredForwardResources.hiZDebugDescriptors.push_back(
                ImGui_ImplVulkan_AddTexture(
                    debugView,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
        }

    }

    void VulkanBackend::destroyHiZDebugDescriptors()
    {
        for (VkDescriptorSet set :
             m_clusteredForwardResources.hiZDebugDescriptors)
        {
            if (set != VK_NULL_HANDLE && m_imguiEnabled)
            {
                ImGui_ImplVulkan_RemoveTexture(set);
            }
        }
        m_clusteredForwardResources.hiZDebugDescriptors.clear();

        for (VkImageView view : m_clusteredForwardResources.hiZDebugViews)
        {
            if (view != VK_NULL_HANDLE)
            {
                vkDestroyImageView(m_device.device(), view, nullptr);
            }
        }
        m_clusteredForwardResources.hiZDebugViews.clear();
    }

    void VulkanBackend::drawHiZDebugWindow()
    {
        if (!m_hiZDebugViewEnabled)
        {
            return;
        }

        ImGui::SetNextWindowSize(ImVec2(360.0f, 260.0f), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("Hi-Z Depth Pyramid", &m_hiZDebugViewEnabled))
        {
            ImGui::End();
            return;
        }

        VulkanGraphResourceEntry* hiZ =
            m_graphResourceRegistry.entry(
                m_clusteredForwardResources.hiZDebugResource);

        if (hiZ)
        {
            ensureHiZDebugDescriptors(*hiZ);
        }

        const auto& descriptors =
            m_clusteredForwardResources.hiZDebugDescriptors;
        if (!hiZ || descriptors.empty())
        {
            ImGui::TextUnformatted("Waiting for Hi-Z pyramid...");
            ImGui::End();
            return;
        }

        const uint32_t mip =
            std::min<uint32_t>(
                m_hiZDebugMip,
                static_cast<uint32_t>(descriptors.size() - 1u));
        const uint32_t mipWidth = std::max(1u, hiZ->width >> mip);
        const uint32_t mipHeight = std::max(1u, hiZ->height >> mip);
        const ImVec2 available = ImGui::GetContentRegionAvail();
        const float maxWidth = std::max(1.0f, available.x);
        const float maxHeight = std::max(1.0f, available.y - 8.0f);
        const float aspect =
            static_cast<float>(mipWidth) / static_cast<float>(mipHeight);
        float width = maxWidth;
        float height = width / aspect;
        if (height > maxHeight)
        {
            height = maxHeight;
            width = height * aspect;
        }

        ImGui::Text("Mip %u / %u", mip, static_cast<uint32_t>(descriptors.size()));
        ImGui::Text(
            "%ux%u",
            mipWidth,
            mipHeight);
        ImGui::Image(
            (ImTextureID)descriptors[mip],
            ImVec2(width, height));
        ImGui::End();
    }

    void VulkanBackend::endDebugGuiFrame()
    {
        if (!m_imguiFrameActive)
        {
            return;
        }

        drawHiZDebugWindow();
        ImGui::Render();
        m_imguiFrameActive = false;
    }


    void VulkanBackend::recordImGui(
        const FrameContext& ctx,
        VkImage swapchainImage,
        std::vector<VkCommandBuffer>& commandBuffers)
    {
        if (!m_imguiEnabled)
        {
            return;
        }

        ImDrawData* drawData = ImGui::GetDrawData();
        if (!drawData || drawData->TotalVtxCount == 0)
        {
            return;
        }

        auto lease =
            m_commandSystem.acquireFrameCommandBuffer(
                static_cast<uint32_t>(ctx.frameIndex % m_frameExecutor.framesInFlight()),
                0);

        VkCommandBuffer cmd = lease.commandBuffer();

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS)
        {
            throw std::runtime_error(
                "Failed to begin Vulkan ImGui command buffer.");
        }

        VulkanGraphResourceEntry* hiZDebugResource = nullptr;
        if (m_hiZDebugViewEnabled)
        {
            hiZDebugResource =
                m_graphResourceRegistry.entry(
                    m_clusteredForwardResources.hiZDebugResource);
        }

        // ImGui is an external consumer of the compiled frame graph. Acquire
        // the pyramid for sampling here, then restore the graph's GENERAL
        // layout below so the next frame starts from its compiled state.
        if (hiZDebugResource && hiZDebugResource->texture.image != VK_NULL_HANDLE)
        {
            transitionImage(
                cmd,
                hiZDebugResource->texture.image,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                0,
                VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0,
                hiZDebugResource->mipLevels);
        }

        transitionImage(
            cmd,
            swapchainImage,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            0,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

        VkRenderingAttachmentInfo colorAttachment{};
        colorAttachment.sType =
            VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttachment.imageView =
            m_swapchain.imageView(m_frameExecutor.currentSwapchainImage());
        colorAttachment.imageLayout =
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingInfo renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderingInfo.renderArea = { {0, 0}, m_swapchain.extent() };
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;

        vkCmdBeginRendering(cmd, &renderingInfo);
        ImGui_ImplVulkan_RenderDrawData(drawData, cmd);
        vkCmdEndRendering(cmd);

        if (hiZDebugResource && hiZDebugResource->texture.image != VK_NULL_HANDLE)
        {
            transitionImage(
                cmd,
                hiZDebugResource->texture.image,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_ACCESS_SHADER_READ_BIT,
                0,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                0,
                hiZDebugResource->mipLevels);
        }

        transitionImage(
            cmd,
            swapchainImage,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            0,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
        {
            throw std::runtime_error(
                "Failed to end Vulkan ImGui command buffer.");
        }

        commandBuffers.push_back(cmd);
    }


    bool VulkanBackend::vsyncEnabled() const
    {
        return m_swapchain.vsyncEnabled();
    }

    void VulkanBackend::setVsyncEnabled(bool enabled)
    {
        if (m_swapchain.setVsyncEnabled(enabled))
        {
            onSwapchainRecreated();
        }
    }

    bool VulkanBackend::clusteredForwardHeatmapEnabled() const
    {
        return m_clusteredForwardHeatmapEnabled;
    }

    void VulkanBackend::setClusteredForwardHeatmapEnabled(bool enabled)
    {
        m_clusteredForwardHeatmapEnabled = enabled;
    }

    bool VulkanBackend::hiZDebugViewEnabled() const
    {
        return m_hiZDebugViewEnabled;
    }

    void VulkanBackend::setHiZDebugViewEnabled(bool enabled)
    {
        if (m_hiZDebugViewEnabled != enabled)
        {
            spdlog::info(
                "[VulkanBackend] Hi-Z debug view {}",
                enabled ? "enabled" : "disabled");
        }
        m_hiZDebugViewEnabled = enabled;
    }

    uint32_t VulkanBackend::hiZDebugMip() const
    {
        return m_hiZDebugMip;
    }

    void VulkanBackend::setHiZDebugMip(uint32_t mip)
    {
        m_hiZDebugMip = mip;
    }

    bool VulkanBackend::supportsAsyncCompute() const
    {
        // A distinct compute queue handle is required to overlap graphics work.
        // When the physical device only offered one queue, computeQueue aliases
        // the graphics queue and there is nothing to overlap, so report false.
        return m_device.computeQueue() != VK_NULL_HANDLE &&
            m_device.computeQueue() != m_device.graphicsQueue();
    }


    VkImageLayout VulkanBackend::usageToLayout(ResourceUsage usage) const
    {
        switch (usage)
        {
        case ResourceUsage::SampledTexture:
            return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        case ResourceUsage::StorageTexture:
            return VK_IMAGE_LAYOUT_GENERAL;

        case ResourceUsage::ColorAttachment:
            return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        case ResourceUsage::DepthAttachment:
            return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        case ResourceUsage::Present:
            return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        case ResourceUsage::TransferDst:
            return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

        case ResourceUsage::TransferSrc:
            return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

        default:
            return VK_IMAGE_LAYOUT_GENERAL;
        }
    }

    VkAccessFlags VulkanBackend::toAccessMask(AccessType type) const
    {
        switch (type)
        {
        case AccessType::Read:
            return VK_ACCESS_SHADER_READ_BIT;

        case AccessType::Write:
            return VK_ACCESS_SHADER_WRITE_BIT;

        case AccessType::ReadWrite:
            return VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        }
        return 0;
    }

    VkAccessFlags VulkanBackend::accessMaskFor(
        ResourceUsage usage,
        AccessType access) const
    {
        if (access == AccessType::ReadWrite)
        {
            return
                accessMaskFor(usage, AccessType::Read) |
                accessMaskFor(usage, AccessType::Write);
        }

        if (access == AccessType::Read)
        {
            switch (usage)
            {
            case ResourceUsage::SampledTexture:
                return VK_ACCESS_SHADER_READ_BIT;

            case ResourceUsage::ColorAttachment:
                return VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;

            case ResourceUsage::DepthAttachment:
                return VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;

            case ResourceUsage::VertexBuffer:
                return VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;

            case ResourceUsage::IndexBuffer:
                return VK_ACCESS_INDEX_READ_BIT;

            case ResourceUsage::ConstantBuffer:
                return VK_ACCESS_UNIFORM_READ_BIT;

            case ResourceUsage::StorageBuffer:
            case ResourceUsage::StorageTexture:
                return VK_ACCESS_SHADER_READ_BIT;

            case ResourceUsage::IndirectArgument:
                return VK_ACCESS_INDIRECT_COMMAND_READ_BIT;

            case ResourceUsage::TransferSrc:
                return VK_ACCESS_TRANSFER_READ_BIT;

            case ResourceUsage::Present:
                return 0;

            default:
                return VK_ACCESS_MEMORY_READ_BIT;
            }
        }

        switch (usage)
        {
        case ResourceUsage::ColorAttachment:
            return VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        case ResourceUsage::DepthAttachment:
            return VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        case ResourceUsage::StorageBuffer:
        case ResourceUsage::StorageTexture:
            return VK_ACCESS_SHADER_WRITE_BIT;

        case ResourceUsage::TransferDst:
            return VK_ACCESS_TRANSFER_WRITE_BIT;

        case ResourceUsage::Present:
            return 0;

        default:
            return VK_ACCESS_MEMORY_WRITE_BIT;
        }
    }

    VkPipelineStageFlags VulkanBackend::pipelineStageFor(
        ResourceUsage usage,
        AccessType access) const
    {
        switch (usage)
        {
        case ResourceUsage::ColorAttachment:
            return VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

        case ResourceUsage::DepthAttachment:
            return
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;

        case ResourceUsage::SampledTexture:
            return VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

        case ResourceUsage::StorageTexture:
        case ResourceUsage::StorageBuffer:
            return VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

        case ResourceUsage::IndirectArgument:
            return VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;

        case ResourceUsage::VertexBuffer:
        case ResourceUsage::IndexBuffer:
            return VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;

        case ResourceUsage::ConstantBuffer:
            return
                VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

        case ResourceUsage::TransferDst:
        case ResourceUsage::TransferSrc:
            return VK_PIPELINE_STAGE_TRANSFER_BIT;

        case ResourceUsage::Present:
            return access == AccessType::Read
                ? VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

        default:
            return VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        }
    }

    void VulkanBackend::onSwapchainRecreated()
    {
        // Bump first so every recreate path (reconcile, out-of-date acquire or
        // present) advances the generation the renderer watches.
        ++m_swapchainGeneration;
        // VulkanSwapchain::recreate performs the single teardown idle wait.
        // Recreate first so all queues are drained before immediate destruction
        // of swapchain-dependent scene and pipeline objects.
        m_swapchain.recreate();
        destroySceneResources();
        m_graphResourceRegistry.reset();
        m_pipelineManager.shutdown();

        if (!m_swapchain.validForRendering())
            return;

        m_frameExecutor.initSwapchainSync();
        m_pipelineManager.init(m_device.device());
        m_gpuScene.init(m_resourceAllocator, m_frameExecutor.framesInFlight());
    }

    TextureFormat VulkanBackend::swapchainTextureFormat() const
    {
        switch (m_swapchain.format())
        {
        case VK_FORMAT_R8G8B8A8_UNORM:
            return TextureFormat::RGBA8_UNorm;
        case VK_FORMAT_R8G8B8A8_SRGB:
            return TextureFormat::RGBA8_SRGB;
        case VK_FORMAT_B8G8R8A8_UNORM:
            return TextureFormat::BGRA8_UNorm;
        case VK_FORMAT_B8G8R8A8_SRGB:
            return TextureFormat::BGRA8_SRGB;
        default:
            return TextureFormat::BGRA8_UNorm;
        }
    }

}
