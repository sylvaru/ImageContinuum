#include "ic/renderer/vulkan_backend/vulkan_backend.h"
#include "ic/core/app_base.h"
#include "ic/core/asset_manager.h"
#include "ic/renderer/pipeline_library.h"
#include "ic/renderer/renderer_specification.h"
#include "ic/renderer/path_tracing/path_trace_scene_builder.h"
#include "ic/interface/window.h"
#include "ic/core/frame_context.h"
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
        constexpr uint32_t DefaultPathTraceMaxBounces = 4;
        constexpr uint32_t DefaultPathTraceSamplesPerPixel = 2;

        struct PathTraceConstants
        {
            uint32_t renderWidth = 0;
            uint32_t renderHeight = 0;
            uint32_t frameIndex = 0;
            uint32_t accumulatedSampleCount = 0;

            float exposure = 1.0f;
            uint32_t resetAccumulation = 1;
            uint32_t maxBounces = 4;
            uint32_t samplesPerPixel = 1;

            uint32_t sceneVertexCount = 0;
            uint32_t sceneMaterialCount = 0;
            uint32_t sceneTriangleCount = 0;
            uint32_t sceneBvhNodeCount = 0;

            uint32_t useSceneGeometry = 0;
            uint32_t padding0 = 0;
            uint32_t padding1 = 0;
            uint32_t padding2 = 0;

            glm::vec4 cameraPositionAndTanHalfFov = glm::vec4(0.0f);
            glm::vec4 cameraForwardAndAspect = glm::vec4(0.0f);
            glm::vec4 cameraRightAndNear = glm::vec4(0.0f);
            glm::vec4 cameraUpAndFar = glm::vec4(0.0f);
        };

        struct TonemapConstants
        {
            uint32_t renderWidth = 0;
            uint32_t renderHeight = 0;
            float exposure = 1.0f;
            uint32_t padding0 = 0;
        };

        static_assert(sizeof(PathTraceConstants) == 128);
        static_assert(sizeof(TonemapConstants) == 16);

        bool matricesDiffer(const glm::mat4& a, const glm::mat4& b)
        {
            constexpr float CameraMatrixEpsilon = 1.0e-5f;
            for (glm::length_t column = 0; column < 4; ++column)
            {
                for (glm::length_t row = 0; row < 4; ++row)
                {
                    if (std::fabs(a[column][row] - b[column][row]) >
                        CameraMatrixEpsilon)
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        void fillPathTraceCameraConstants(
            const SceneCameraView& camera,
            uint32_t width,
            uint32_t height,
            PathTraceConstants& constants)
        {
            const float fallbackAspect =
                height == 0u
                    ? 1.0f
                    : static_cast<float>(width) / static_cast<float>(height);

            if (camera.valid != 0u)
            {
                const glm::mat4 inverseView = glm::inverse(camera.view);
                const glm::vec3 right =
                    glm::normalize(glm::vec3(inverseView[0]));
                const glm::vec3 up =
                    glm::normalize(glm::vec3(inverseView[1]));
                const glm::vec3 forward =
                    glm::normalize(-glm::vec3(inverseView[2]));
                const float aspect =
                    camera.aspectRatio > 0.0f
                        ? camera.aspectRatio
                        : fallbackAspect;
                const float tanHalfFov =
                    std::tan(camera.verticalFovRadians * 0.5f);

                constants.cameraPositionAndTanHalfFov =
                    glm::vec4(camera.position, tanHalfFov);
                constants.cameraForwardAndAspect =
                    glm::vec4(forward, aspect);
                constants.cameraRightAndNear =
                    glm::vec4(right, camera.nearPlane);
                constants.cameraUpAndFar =
                    glm::vec4(up, camera.farPlane);
                return;
            }

            const glm::vec3 origin(0.0f, 1.05f, -3.25f);
            const glm::vec3 target(0.0f, 0.95f, 0.95f);
            const glm::vec3 forward = glm::normalize(target - origin);
            const glm::vec3 right =
                glm::normalize(
                    glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
            const glm::vec3 up = glm::normalize(glm::cross(right, forward));

            constants.cameraPositionAndTanHalfFov =
                glm::vec4(origin, std::tan(glm::radians(39.0f) * 0.5f));
            constants.cameraForwardAndAspect =
                glm::vec4(forward, fallbackAspect);
            constants.cameraRightAndNear =
                glm::vec4(right, 0.1f);
            constants.cameraUpAndFar =
                glm::vec4(up, 100.0f);
        }

        bool planUsesPathTracing(const CompiledGraphPlan& plan)
        {
            for (const auto& payload : plan.payloads)
            {
                if (std::get_if<PathTracePassData>(&payload))
                {
                    return true;
                }
            }

            return false;
        }
    }

	void VulkanBackend::initialize(
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

		m_swapchain.init(
			m_adapter,
			m_device,
			m_platform.surface(),
			window);

        const uint32_t framesInFlight =
            spec.framesInFlight == 0 ? 1 : spec.framesInFlight;

        const uint32_t workerSlots =
            workerCount == 0 ? 1 : workerCount;
        m_workerSlots = workerSlots;

        initFrameSync(spec);

		m_commandSystem.init(
			m_device.device(),
			m_adapter.info().queueFamilies.graphics,
			framesInFlight,
			workerSlots);

        m_descriptorSystem.init(
            m_device.device(),
            m_device.info());

        m_pipelineManager.init(m_device.device());
        m_sceneFrameResources.resize(framesInFlight);
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
        m_pipelineManager.shutdown();

        destroySwapchainSync();
        destroyFrameSync();

        m_descriptorSystem.shutdown();
        m_resourceAllocator.shutdown();
        m_commandSystem.shutdown();
        m_swapchain.shutdown();
        m_device.shutdown();
        m_platform.shutdown();
        m_instance.shutdown();

		spdlog::info("[VulkanBackend] Shutdown");
	}

    void VulkanBackend::execute(
        const CompiledGraphPlan& plan,
        const FrameContext& ctx,
        const SceneRenderView& scene)
    {
        if (planUsesPathTracing(plan) &&
            m_device.device() != VK_NULL_HANDLE)
        {
            vkDeviceWaitIdle(m_device.device());
        }

        const uint32_t frameSlot =
            static_cast<uint32_t>(ctx.frameIndex % m_frameSync.size());

        auto& frameSync =
            m_frameSync[frameSlot];

        // CPU/GPU sync (frame pacing)
        vkWaitForFences(
            m_device.device(),
            1,
            &frameSync.inFlightFence,
            VK_TRUE,
            UINT64_MAX);

        // Acquire swapchain image
        uint32_t imageIndex =
            m_swapchain.acquireNextImage(
                frameSync.imageAvailable);

        if (imageIndex == UINT32_MAX)
        {
            onSwapchainRecreated();
            return;
        }

        m_currentSwapchainImage = imageIndex;

        VkImage swapchainImage =
            m_swapchain.image(imageIndex);

        const VkImageLayout swapchainInitialLayout =
            imageIndex < m_swapchainImageLayouts.size()
                ? m_swapchainImageLayouts[imageIndex]
                : VK_IMAGE_LAYOUT_UNDEFINED;

        if (m_imagesInFlight[imageIndex] != VK_NULL_HANDLE)
        {
            vkWaitForFences(
                m_device.device(),
                1,
                &m_imagesInFlight[imageIndex],
                VK_TRUE,
                UINT64_MAX);
        }

        m_imagesInFlight[imageIndex] =
            frameSync.inFlightFence;

        vkResetFences(
            m_device.device(),
            1,
            &frameSync.inFlightFence);

        m_commandSystem.beginFrame(frameSlot);

        std::vector<VkCommandBuffer> commandBuffers;
        executeGraph(
            plan,
            ctx,
            scene,
            swapchainImage,
            swapchainInitialLayout,
            commandBuffers);
        recordImGui(
            ctx,
            swapchainImage,
            commandBuffers);

        VkSemaphore renderFinished =
            m_imageRenderFinished[imageIndex];

        // Submit
        submitFrame(
            commandBuffers,
            frameSync,
            renderFinished);

        if (imageIndex < m_swapchainImageLayouts.size())
        {
            m_swapchainImageLayouts[imageIndex] =
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        }

        // Present
        const bool presented =
            m_swapchain.present(
            m_device.presentQueue(),
            imageIndex,
            renderFinished);

        if (!presented)
        {
            onSwapchainRecreated();
        }
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

    void VulkanBackend::endDebugGuiFrame()
    {
        if (!m_imguiFrameActive)
        {
            return;
        }

        ImGui::Render();
        m_imguiFrameActive = false;
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
                static_cast<uint32_t>(ctx.frameIndex % m_frameSync.size()),
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
            m_swapchain.imageView(m_currentSwapchainImage);
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

    void VulkanBackend::submitFrame(
        std::span<const VkCommandBuffer> commandBuffers,
        FrameSync& sync,
        VkSemaphore renderFinished)
    {
        VkPipelineStageFlags waitStage =
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &sync.imageAvailable;
        submit.pWaitDstStageMask = &waitStage;

        submit.commandBufferCount =
            static_cast<uint32_t>(commandBuffers.size());
        submit.pCommandBuffers = commandBuffers.data();

        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &renderFinished;

        VkResult result =
            vkQueueSubmit(
            m_device.graphicsQueue(),
            1,
            &submit,
            sync.inFlightFence);

        if (result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to submit Vulkan frame.");
        }
    }

    void VulkanBackend::executeGraph(
        const CompiledGraphPlan& plan,
        const FrameContext& ctx,
        const SceneRenderView& scene,
        VkImage swapchainImage,
        VkImageLayout swapchainInitialLayout,
        std::vector<VkCommandBuffer>& commandBuffers)
    {
        auto recordNode =
            [&](GraphNodeId nodeId, uint32_t workerIndex)
            {
                auto lease =
                    m_commandSystem.acquireFrameCommandBuffer(
                        static_cast<uint32_t>(ctx.frameIndex % m_frameSync.size()),
                        workerIndex);

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

                dispatchNode(
                    plan,
                    node,
                    ctx,
                    scene,
                    cmd,
                    swapchainImage);

                if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
                {
                    throw std::runtime_error(
                        "Failed to end Vulkan command buffer.");
                }

                return cmd;
            };

        if (plan.executionLevels.empty())
        {
            for (uint32_t i = 0; i < plan.executionOrder.size(); ++i)
            {
                commandBuffers.push_back(
                    recordNode(plan.executionOrder[i], i % m_workerSlots));
            }

            return;
        }

        for (const ExecutionLevel& level : plan.executionLevels)
        {
            for (uint32_t i = 0; i < level.nodeCount; ++i)
            {
                const GraphNodeId nodeId =
                    plan.executionLevelNodes[level.firstNode + i];

                commandBuffers.push_back(
                    recordNode(nodeId, i % m_workerSlots));
            }
        }
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
                    swapchainInitialLayout);
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

            recordBarrier(
                cmd,
                plan.barriers[barrierIndex],
                plan.resources,
                swapchainImage,
                swapchainInitialLayout);
        }
    }

    void VulkanBackend::recordBarrier(
        VkCommandBuffer cmd,
        const ResourceBarrier& barrier,
        std::span<const GraphResource> resources,
        VkImage swapchainImage,
        VkImageLayout swapchainInitialLayout)
    {

        VkImageMemoryBarrier vkBarrier{};
        vkBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;

        const GraphResource& resource =
            resources[barrier.resource];

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
            // TODO
            return;
        }
        vkBarrier.image = image;

        const bool isExternalFirstUse =
            isSwapchainImage &&
            barrier.fromNode == barrier.toNode;

        vkBarrier.oldLayout = isExternalFirstUse
            ? swapchainInitialLayout
            : usageToLayout(barrier.oldUsage);

        vkBarrier.newLayout = usageToLayout(barrier.newUsage);

        vkBarrier.srcAccessMask =
            vkBarrier.oldLayout == VK_IMAGE_LAYOUT_UNDEFINED
                ? 0
                : accessMaskFor(
                    barrier.oldUsage,
                    barrier.fromAccess);

        vkBarrier.dstAccessMask =
            accessMaskFor(
                barrier.newUsage,
                barrier.toAccess);

        vkBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vkBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        vkBarrier.srcQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;

        vkBarrier.dstQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;

        vkBarrier.subresourceRange.aspectMask =
            VK_IMAGE_ASPECT_COLOR_BIT;

        vkBarrier.subresourceRange.baseMipLevel = 0;
        vkBarrier.subresourceRange.levelCount = 1;
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

        VkPipelineStageFlags dstStage =
            pipelineStageFor(
                barrier.newUsage,
                barrier.toAccess);

        if (barrier.newUsage == ResourceUsage::Present)
        {
            dstStage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        }

        vkCmdPipelineBarrier(
            cmd,
            srcStage,
            dstStage,
            0,
            0, nullptr,
            0, nullptr,
            1, &vkBarrier);
    }

    void VulkanBackend::dispatchNode(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node,
        const FrameContext& ctx,
        const SceneRenderView& scene,
        VkCommandBuffer cmd,
        [[maybe_unused]] VkImage swapchainImage)
    {
        switch (node.type)
        {
        case GraphNodeType::Graphics:
            executeGraphicsNode(plan, node, ctx, scene, cmd, swapchainImage);
            break;

        case GraphNodeType::Compute:
            executeComputeNode(plan, node, ctx, scene, cmd);
            break;

        case GraphNodeType::Transfer:
            executeTransferNode(plan, node, ctx, cmd, swapchainImage);
            break;

        case GraphNodeType::Present:
            break;
        }
    }

    void VulkanBackend::executeGraphicsNode(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node,
        const FrameContext& ctx,
        const SceneRenderView& scene,
        VkCommandBuffer cmd,
        [[maybe_unused]] VkImage swapchainImage)
    {
        ensureDepthTarget();

        const GraphicsPipelineHandle pipelineHandle =
            pipelineForNode(plan, node);
        VulkanGraphicsPipeline* pipeline =
            m_pipelineManager.graphicsPipeline(pipelineHandle);
        if (!pipeline)
        {
            return;
        }

        const bool hasColorTarget =
            pipeline->desc.colorAttachmentCount > 0;

        VkClearValue clear{};
        clear.color = { { 0.02f, 0.02f, 0.025f, 1.0f } };

        VkImageMemoryBarrier depthBarrier{};
        depthBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        depthBarrier.oldLayout = m_depthLayout;
        depthBarrier.newLayout =
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthBarrier.srcAccessMask = 0;
        depthBarrier.dstAccessMask =
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        depthBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        depthBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        depthBarrier.image = m_depthTexture.image;
        depthBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        depthBarrier.subresourceRange.baseMipLevel = 0;
        depthBarrier.subresourceRange.levelCount = 1;
        depthBarrier.subresourceRange.baseArrayLayer = 0;
        depthBarrier.subresourceRange.layerCount = 1;

        if (m_depthLayout !=
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        {
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
        colorAttachment.imageView = m_swapchain.imageView(m_currentSwapchainImage);
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.clearValue = clear;

        VkClearValue depthClear{};
        depthClear.depthStencil = { 1.0f, 0 };

        VkRenderingAttachmentInfo depthAttachment{};
        depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depthAttachment.imageView = m_depthImageView;
        depthAttachment.imageLayout =
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthAttachment.loadOp =
            hasColorTarget
                ? VK_ATTACHMENT_LOAD_OP_LOAD
                : VK_ATTACHMENT_LOAD_OP_CLEAR;
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

        vkCmdBeginRendering(cmd, &renderingInfo);

        if (prepareSceneResources(ctx, scene, pipelineHandle) &&
            !m_preparedScene.draws.empty())
        {
            const std::vector<DrawItem>& draws =
                m_preparedScene.draws;
            const VkExtent2D extent = m_swapchain.extent();

            VkViewport viewport{};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = static_cast<float>(extent.width);
            viewport.height = static_cast<float>(extent.height);
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;

            VkRect2D scissor{};
            scissor.offset = { 0, 0 };
            scissor.extent = extent;

            vkCmdSetViewport(cmd, 0, 1, &viewport);
            vkCmdSetScissor(cmd, 0, 1, &scissor);

            const uint32_t frameSlot =
                static_cast<uint32_t>(
                    ctx.frameIndex % m_sceneFrameResources.size());
            FrameSceneResources& frameResources =
                m_sceneFrameResources[frameSlot];

            vkCmdBindPipeline(
                cmd,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipeline->pipeline);

            vkCmdBindDescriptorSets(
                cmd,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipeline->pipelineLayout,
                0,
                1,
                &frameResources.descriptorSet,
                0,
                nullptr);

            UploadedModel* boundModel = nullptr;
            for (const DrawItem& draw : draws)
            {
                if (!draw.model ||
                    draw.meshIndex >= draw.model->meshes.size())
                {
                    continue;
                }

                const GpuMesh& mesh =
                    draw.model->meshes[draw.meshIndex];
                if (mesh.indexCount == 0)
                {
                    continue;
                }

                if (boundModel != draw.model)
                {
                    VkDeviceSize offset = 0;
                    vkCmdBindVertexBuffers(
                        cmd,
                        0,
                        1,
                        &draw.model->vertexBuffer.buffer,
                        &offset);
                    vkCmdBindIndexBuffer(
                        cmd,
                        draw.model->indexBuffer.buffer,
                        0,
                        VK_INDEX_TYPE_UINT32);
                    boundModel = draw.model;
                }

                DrawConstants constants{};
                constants.objectIndex = draw.objectIndex;
                constants.meshIndex = draw.meshIndex;
                constants.materialIndex = draw.materialIndex;

                vkCmdPushConstants(
                    cmd,
                    pipeline->pipelineLayout,
                    VK_SHADER_STAGE_VERTEX_BIT |
                        VK_SHADER_STAGE_FRAGMENT_BIT,
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

        vkCmdEndRendering(cmd);

    }

    void VulkanBackend::executeComputeNode(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node,
        [[maybe_unused]] const FrameContext& ctx,
        [[maybe_unused]] const SceneRenderView& scene,
        [[maybe_unused]] VkCommandBuffer cmd)
    {
        if (node.payloadIndex >= plan.payloads.size())
        {
            return;
        }

        if (std::get_if<PathTracePassData>(&plan.payloads[node.payloadIndex]))
        {
            executePathTraceNode(plan, node, ctx, scene, cmd);
            return;
        }

        if (std::get_if<TonemapPassData>(&plan.payloads[node.payloadIndex]))
        {
            executeTonemapNode(plan, node, ctx, cmd);
            return;
        }

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
            vkCmdBindDescriptorSets(
                cmd,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline->pipelineLayout,
                0,
                1,
                &m_computeTestDescriptorSet,
                0,
                nullptr);
        }

        vkCmdDispatch(
            cmd,
            pass->groupCountX,
            pass->groupCountY,
            pass->groupCountZ);

        if (pipeline->desc.bindingLayout ==
            PipelineBindingLayoutKind::ComputeStorageBuffer)
        {
            VkBufferMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask =
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = m_computeTestBuffer.buffer;
            barrier.offset = 0;
            barrier.size = m_computeTestBuffer.size;

            vkCmdPipelineBarrier(
                cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr,
                1, &barrier,
                0, nullptr);
        }
    }

    void VulkanBackend::executePathTraceNode(
        const CompiledGraphPlan& plan,
        const ExecutionNode& node,
        const FrameContext& ctx,
        const SceneRenderView& scene,
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
        ensurePathTraceSceneResources(ctx, scene);
        updatePathTraceDescriptors(pipeline, nullptr);

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

        PathTraceConstants constants{};
        constants.renderWidth = m_pathTraceResources.width;
        constants.renderHeight = m_pathTraceResources.height;
        constants.frameIndex = static_cast<uint32_t>(ctx.frameIndex);
        constants.accumulatedSampleCount =
            m_pathTraceResources.accumulatedSampleCount;
        constants.exposure = 1.0f;
        constants.resetAccumulation =
            m_pathTraceResources.resetAccumulation ? 1u : 0u;
        constants.maxBounces = DefaultPathTraceMaxBounces;
        constants.samplesPerPixel = DefaultPathTraceSamplesPerPixel;
        constants.sceneVertexCount =
            m_pathTraceResources.sceneVertexCount;
        constants.sceneMaterialCount =
            m_pathTraceResources.sceneMaterialCount;
        constants.sceneTriangleCount =
            m_pathTraceResources.sceneTriangleCount;
        constants.sceneBvhNodeCount =
            m_pathTraceResources.sceneBvhNodeCount;
        constants.useSceneGeometry =
            m_pathTraceResources.sceneTriangleCount != 0u &&
            m_pathTraceResources.sceneBvhNodeCount != 0u
                ? 1u
                : 0u;
        fillPathTraceCameraConstants(
            scene.camera,
            m_pathTraceResources.width,
            m_pathTraceResources.height,
            constants);

        VulkanBuffer& pathTraceConstants =
            m_pathTraceResources.pathTraceConstants[frameSlot];

        std::memcpy(
            pathTraceConstants.mapped,
            &constants,
            sizeof(constants));
        m_resourceAllocator.flush(
            pathTraceConstants,
            0,
            sizeof(constants));

        vkCmdBindPipeline(
            cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline->pipeline);
        vkCmdBindDescriptorSets(
            cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline->pipelineLayout,
            0,
            1,
            &descriptorSet,
            0,
            nullptr);

        const uint32_t groupCountX =
            (m_pathTraceResources.width + 7u) / 8u;
        const uint32_t groupCountY =
            (m_pathTraceResources.height + 7u) / 8u;
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);

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
        updatePathTraceDescriptors(nullptr, pipeline);

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

        TonemapConstants constants{};
        constants.renderWidth = m_pathTraceResources.width;
        constants.renderHeight = m_pathTraceResources.height;
        constants.exposure = 1.0f;

        VulkanBuffer& tonemapConstants =
            m_pathTraceResources.tonemapConstants[frameSlot];

        std::memcpy(
            tonemapConstants.mapped,
            &constants,
            sizeof(constants));
        m_resourceAllocator.flush(
            tonemapConstants,
            0,
            sizeof(constants));

        vkCmdBindPipeline(
            cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline->pipeline);
        vkCmdBindDescriptorSets(
            cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline->pipelineLayout,
            0,
            1,
            &descriptorSet,
            0,
            nullptr);

        const uint32_t groupCountX =
            (m_pathTraceResources.width + 7u) / 8u;
        const uint32_t groupCountY =
            (m_pathTraceResources.height + 7u) / 8u;
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);
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

        if (pass->name == "PathTracer.CopyToBackBuffer" &&
            m_pathTraceResources.tonemap &&
            swapchainImage != VK_NULL_HANDLE)
        {
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

            VkImageCopy copy{};
            copy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copy.srcSubresource.layerCount = 1;
            copy.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copy.dstSubresource.layerCount = 1;
            copy.extent = {
                m_pathTraceResources.width,
                m_pathTraceResources.height,
                1
            };

            vkCmdCopyImage(
                cmd,
                m_pathTraceResources.tonemap.image,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                swapchainImage,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &copy);
        }
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

        auto it = m_pipelineHandles.find(pass->pipeline);
        if (it != m_pipelineHandles.end())
        {
            return it->second;
        }

        GraphicsPipelineDesc desc =
            m_pipelineLibrary->resolveGraphics(
                pass->pipeline,
                RendererBackendType::Vulkan,
                swapchainTextureFormat());

        GraphicsPipelineHandle handle =
            m_pipelineManager.requestGraphicsPipeline(desc);
        m_pipelineHandles.emplace(pass->pipeline, handle);
        return handle;
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

        auto it = m_computePipelineHandles.find(pipelineId);
        if (it != m_computePipelineHandles.end())
        {
            return it->second;
        }

        ComputePipelineDesc desc =
            m_pipelineLibrary->resolveCompute(
                pipelineId,
                RendererBackendType::Vulkan);

        ComputePipelineHandle handle =
            m_pipelineManager.requestComputePipeline(desc);
        m_computePipelineHandles.emplace(pipelineId, handle);
        return handle;
    }

    void VulkanBackend::destroySceneResources()
    {
        destroyDepthTarget();
        destroyPathTraceResources();

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

        for (FrameSceneResources& resources : m_sceneFrameResources)
        {
            m_resourceAllocator.destroyBuffer(resources.frameConstants);
            m_resourceAllocator.destroyBuffer(resources.objects);
            m_resourceAllocator.destroyBuffer(resources.materials);

            if (resources.descriptorPool != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorPool(
                    m_device.device(),
                    resources.descriptorPool,
                    nullptr);
            }

            resources = {};
        }
        m_sceneFrameResources.clear();
        m_preparedScene = {};

        m_pipelineHandles.clear();
        m_computePipelineHandles.clear();
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
            static_cast<uint32_t>(std::max<size_t>(1, m_frameSync.size()));
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

        VkDescriptorPoolSize poolSizes[4]{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = 2 * frameCount;
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        poolSizes[1].descriptorCount = 2 * frameCount;
        poolSizes[2].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        poolSizes[2].descriptorCount = frameCount;
        poolSizes[3].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[3].descriptorCount = 4 * frameCount;

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = 2 * frameCount;
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

        const bool retryPendingScene =
            m_pathTraceResources.sceneTriangleCount == 0u &&
            !scene.models.empty();
        const bool retryThisFrame =
            retryPendingScene &&
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
                *ctx.services->assetManager);
        if (sceneData.triangles.empty() &&
            m_pathTraceResources.sceneVertices)
        {
            return;
        }

        uploadPathTraceScene(sceneData);

        m_pathTraceResources.sceneVersion = scene.sceneVersion;
        m_pathTraceResources.accumulatedSampleCount = 0;
        m_pathTraceResources.resetAccumulation = true;
    }

    void VulkanBackend::uploadPathTraceScene(
        const PathTraceSceneData& sceneData)
    {
        if (m_device.device() != VK_NULL_HANDLE &&
            (m_pathTraceResources.sceneVertices ||
                std::ranges::any_of(
                    m_pathTraceResources.pathTraceDescriptorSets,
                    [](VkDescriptorSet set)
                    {
                        return set != VK_NULL_HANDLE;
                    })))
        {
            vkDeviceWaitIdle(m_device.device());
        }

        destroyPathTraceSceneResources();

        m_pathTraceResources.sceneVertexCount =
            static_cast<uint32_t>(sceneData.vertices.size());
        m_pathTraceResources.sceneMaterialCount =
            static_cast<uint32_t>(sceneData.materials.size());
        m_pathTraceResources.sceneTriangleCount =
            static_cast<uint32_t>(sceneData.triangles.size());
        m_pathTraceResources.sceneBvhNodeCount =
            static_cast<uint32_t>(sceneData.bvhNodes.size());

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
            m_commandSystem.immediateSubmit(
                m_device.graphicsQueue(),
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
                        barrier.dstAccessMask =
                            VK_ACCESS_SHADER_READ_BIT;
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
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
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
                m_resourceAllocator.destroyBuffer(upload.staging);
            }
        }

        m_pathTraceResources.pathTraceDescriptorsDirty = true;
    }

    void VulkanBackend::destroyPathTraceSceneResources()
    {
        m_resourceAllocator.destroyBuffer(
            m_pathTraceResources.sceneVertices);
        m_resourceAllocator.destroyBuffer(
            m_pathTraceResources.sceneMaterials);
        m_resourceAllocator.destroyBuffer(
            m_pathTraceResources.sceneTriangles);
        m_resourceAllocator.destroyBuffer(
            m_pathTraceResources.sceneBvhNodes);

        m_pathTraceResources.sceneVertexCount = 0;
        m_pathTraceResources.sceneMaterialCount = 0;
        m_pathTraceResources.sceneTriangleCount = 0;
        m_pathTraceResources.sceneBvhNodeCount = 0;
    }

    void VulkanBackend::destroyPathTraceResources()
    {
        VkDevice device = m_device.device();

        if (device != VK_NULL_HANDLE)
        {
            if (m_pathTraceResources.accumulationView != VK_NULL_HANDLE)
            {
                vkDestroyImageView(
                    device,
                    m_pathTraceResources.accumulationView,
                    nullptr);
            }

            if (m_pathTraceResources.tonemapView != VK_NULL_HANDLE)
            {
                vkDestroyImageView(
                    device,
                    m_pathTraceResources.tonemapView,
                    nullptr);
            }

            if (m_pathTraceResources.descriptorPool != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorPool(
                    device,
                    m_pathTraceResources.descriptorPool,
                    nullptr);
            }
        }

        destroyPathTraceSceneResources();

        m_resourceAllocator.destroyTexture(
            m_pathTraceResources.accumulation);
        m_resourceAllocator.destroyTexture(
            m_pathTraceResources.tonemap);
        for (VulkanBuffer& buffer : m_pathTraceResources.pathTraceConstants)
        {
            m_resourceAllocator.destroyBuffer(buffer);
        }
        for (VulkanBuffer& buffer : m_pathTraceResources.tonemapConstants)
        {
            m_resourceAllocator.destroyBuffer(buffer);
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
        const VulkanComputePipeline* tonemapPipeline)
    {
        if (m_pathTraceResources.descriptorPool == VK_NULL_HANDLE)
        {
            return;
        }

        if (pathTracePipeline)
        {
            bool updateDescriptors =
                m_pathTraceResources.pathTraceDescriptorsDirty;

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
                const uint32_t setCount =
                    static_cast<uint32_t>(
                        m_pathTraceResources.pathTraceDescriptorSets.size());
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

                if (vkAllocateDescriptorSets(
                        m_device.device(),
                        &allocateInfo,
                        m_pathTraceResources.pathTraceDescriptorSets.data()) !=
                    VK_SUCCESS)
                {
                    throw std::runtime_error(
                        "Failed to allocate Vulkan path trace descriptor set.");
                }

                updateDescriptors = true;
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

            VkDescriptorBufferInfo* sceneInfos[4] =
            {
                &vertexInfo,
                &materialInfo,
                &triangleInfo,
                &bvhInfo
            };

            std::vector<VkDescriptorBufferInfo> constantsInfos(
                m_pathTraceResources.pathTraceConstants.size());
            std::vector<VkWriteDescriptorSet> writes;
            writes.reserve(
                m_pathTraceResources.pathTraceDescriptorSets.size() * 6u);

            for (size_t setIndex = 0;
                 setIndex < m_pathTraceResources.pathTraceDescriptorSets.size();
                 ++setIndex)
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
            }

            vkUpdateDescriptorSets(
                m_device.device(),
                static_cast<uint32_t>(writes.size()),
                writes.data(),
                0,
                nullptr);

            m_pathTraceResources.pathTraceDescriptorsDirty = false;
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

                if (vkAllocateDescriptorSets(
                        m_device.device(),
                        &allocateInfo,
                        m_pathTraceResources.tonemapDescriptorSets.data()) !=
                    VK_SUCCESS)
                {
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
                .usage = TextureUsageFlags::DepthAttachment,
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

    VulkanBackend::UploadedModel* VulkanBackend::requestModel(
        AssetHandle handle,
        const AssetManager& assets)
    {
        if (!handle)
        {
            return nullptr;
        }

        auto [it, inserted] =
            m_uploadedModels.try_emplace(handle);
        UploadedModel& uploaded = it->second;
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

        m_commandSystem.immediateSubmit(
            m_device.graphicsQueue(),
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
                barriers[0].dstAccessMask =
                    VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
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
                barriers[1].dstAccessMask =
                    VK_ACCESS_INDEX_READ_BIT;
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
                    VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                    0,
                    0,
                    nullptr,
                    2,
                    barriers,
                    0,
                    nullptr);
            });

        m_resourceAllocator.destroyBuffer(vertexStaging);
        m_resourceAllocator.destroyBuffer(indexStaging);

        uploaded.materials.reserve(
            std::max<size_t>(1, model->materials.size()));
        for (const MaterialAsset& src : model->materials)
        {
            GpuMaterialData dst{};
            dst.baseColorFactor = src.baseColorFactor;
            dst.metallicFactor = src.metallicFactor;
            dst.roughnessFactor = src.roughnessFactor;
            dst.alphaCutoff = src.alphaCutoff;
            dst.flags =
                (src.doubleSided ? 1u : 0u) |
                (src.alphaBlend ? 2u : 0u) |
                (src.alphaMask ? 4u : 0u);
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
        GraphicsPipelineHandle pipelineHandle)
    {
        if (m_preparedScene.frameIndex == ctx.frameIndex &&
            m_preparedScene.valid)
        {
            if (m_sceneFrameResources.empty())
            {
                return false;
            }

            const uint32_t frameSlot =
                static_cast<uint32_t>(
                    ctx.frameIndex % m_sceneFrameResources.size());
            FrameSceneResources& frameResources =
                m_sceneFrameResources[frameSlot];
            if (frameResources.descriptorSet == VK_NULL_HANDLE)
            {
                updateFrameDescriptors(frameResources, pipelineHandle);
            }
            return true;
        }

        m_preparedScene.frameIndex = ctx.frameIndex;
        m_preparedScene.valid = false;
        m_preparedScene.draws.clear();
        m_preparedScene.objects.clear();
        m_preparedScene.materials.clear();
        m_preparedScene.materialOffsets.clear();

        if (!ctx.services ||
            !ctx.services->assetManager ||
            scene.camera.valid == 0 ||
            m_sceneFrameResources.empty())
        {
            return false;
        }

        AssetManager& assets = *ctx.services->assetManager;
        std::vector<DrawItem>& draws = m_preparedScene.draws;
        std::vector<GpuObjectData>& objects = m_preparedScene.objects;
        std::vector<GpuMaterialData>& materials = m_preparedScene.materials;
        auto& materialOffsets = m_preparedScene.materialOffsets;

        m_uploadedModels.reserve(
            m_uploadedModels.size() + scene.models.size());
        objects.reserve(scene.models.size());
        materialOffsets.reserve(scene.models.size());

        for (const SceneModelRenderItem& item : scene.models)
        {
            UploadedModel* model = requestModel(item.model, assets);
            if (!model)
            {
                continue;
            }

            uint32_t materialOffset = 0;
            if (auto it = materialOffsets.find(item.model);
                it != materialOffsets.end())
            {
                materialOffset = it->second;
            }
            else
            {
                materialOffset = static_cast<uint32_t>(materials.size());
                materialOffsets.emplace(item.model, materialOffset);
                materials.insert(
                    materials.end(),
                    model->materials.begin(),
                    model->materials.end());
            }

            for (uint32_t meshIndex = 0;
                 meshIndex < model->meshes.size();
                 ++meshIndex)
            {
                const GpuMesh& mesh = model->meshes[meshIndex];
                const glm::mat4 meshWorld =
                    meshIndex < model->meshTransforms.size()
                        ? item.world * model->meshTransforms[meshIndex]
                        : item.world;

                const uint32_t objectIndex =
                    static_cast<uint32_t>(objects.size());

                GpuObjectData object{};
                object.world = meshWorld;
                object.inverseTransposeWorld =
                    glm::inverseTranspose(meshWorld);
                objects.push_back(object);

                DrawItem draw{};
                draw.model = model;
                draw.objectIndex = objectIndex;
                draw.meshIndex = meshIndex;
                draw.materialIndex = materialOffset + mesh.materialIndex;
                draws.push_back(draw);
            }
        }

        if (draws.empty())
        {
            return false;
        }

        if (materials.empty())
        {
            materials.push_back(GpuMaterialData{});
        }

        const uint32_t frameSlot =
            static_cast<uint32_t>(
                ctx.frameIndex % m_sceneFrameResources.size());
        FrameSceneResources& frameResources =
            m_sceneFrameResources[frameSlot];

        if (!frameResources.frameConstants)
        {
            frameResources.frameConstants =
                m_resourceAllocator.createBuffer({
                    .size = sizeof(GpuFrameData),
                    .usage = BufferUsageFlags::Constant,
                    .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                    .mappedAtCreation = true,
                    .debugName = "Vulkan frame constants"
                });
        }

        bool descriptorsDirty = false;
        if (objects.size() > frameResources.objectCapacity)
        {
            m_resourceAllocator.destroyBuffer(frameResources.objects);
            frameResources.objects =
                m_resourceAllocator.createBuffer({
                    .size = objects.size() * sizeof(GpuObjectData),
                    .usage = BufferUsageFlags::Storage,
                    .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                    .mappedAtCreation = true,
                    .debugName = "Vulkan object data"
                });
            frameResources.objectCapacity =
                static_cast<uint32_t>(objects.size());
            descriptorsDirty = true;
        }

        if (materials.size() > frameResources.materialCapacity)
        {
            m_resourceAllocator.destroyBuffer(frameResources.materials);
            frameResources.materials =
                m_resourceAllocator.createBuffer({
                    .size = materials.size() * sizeof(GpuMaterialData),
                    .usage = BufferUsageFlags::Storage,
                    .memoryUsage = ResourceMemoryUsage::CpuToGpu,
                    .mappedAtCreation = true,
                    .debugName = "Vulkan material data"
                });
            frameResources.materialCapacity =
                static_cast<uint32_t>(materials.size());
            descriptorsDirty = true;
        }

        if (descriptorsDirty ||
            frameResources.descriptorSet == VK_NULL_HANDLE)
        {
            updateFrameDescriptors(frameResources, pipelineHandle);
        }

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

        std::memcpy(
            frameResources.frameConstants.mapped,
            &frameData,
            sizeof(frameData));
        std::memcpy(
            frameResources.objects.mapped,
            objects.data(),
            objects.size() * sizeof(GpuObjectData));
        std::memcpy(
            frameResources.materials.mapped,
            materials.data(),
            materials.size() * sizeof(GpuMaterialData));

        m_resourceAllocator.flush(
            frameResources.frameConstants,
            0,
            sizeof(frameData));
        m_resourceAllocator.flush(
            frameResources.objects,
            0,
            objects.size() * sizeof(GpuObjectData));
        m_resourceAllocator.flush(
            frameResources.materials,
            0,
            materials.size() * sizeof(GpuMaterialData));

        m_preparedScene.valid = true;
        return true;
    }

    void VulkanBackend::updateFrameDescriptors(
        FrameSceneResources& resources,
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

        VkDescriptorPoolSize poolSizes[2]{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = 1;
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[1].descriptorCount = 2;

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

        VkWriteDescriptorSet writes[3]{};
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

        vkUpdateDescriptorSets(
            m_device.device(),
            static_cast<uint32_t>(std::size(writes)),
            writes,
            0,
            nullptr);
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

    VkImageLayout VulkanBackend::getOrInitImageLayout(VkImage image)
    {
        auto it = m_imageStates.find(image);
        if (it != m_imageStates.end())
            return it->second.layout;

        m_imageStates[image] = {
            .layout = VK_IMAGE_LAYOUT_UNDEFINED,
            .access = AccessType::Read
        };

        return VK_IMAGE_LAYOUT_UNDEFINED;
    }

    void VulkanBackend::transitionImage(
        VkCommandBuffer cmd,
        VkImage image,
        VkImageLayout oldLayout,
        VkImageLayout newLayout,
        VkAccessFlags srcAccess,
        VkAccessFlags dstAccess,
        VkPipelineStageFlags srcStage,
        VkPipelineStageFlags dstStage)
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;

        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;

        barrier.srcAccessMask = srcAccess;
        barrier.dstAccessMask = dstAccess;

        barrier.image = image;

        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;

        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        vkCmdPipelineBarrier(
            cmd,
            srcStage,
            dstStage,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier);

        m_imageStates[image].layout = newLayout;
        m_imageStates[image].access = (dstAccess != 0)
            ? AccessType::Write
            : AccessType::Read;
    }

    void VulkanBackend::initFrameSync(const RendererSpecification& spec)
    {
        const uint32_t framesInFlight =
            spec.framesInFlight == 0 ? 1 : spec.framesInFlight;

        m_frameSync.resize(framesInFlight);

        for (auto& fs : m_frameSync)
        {
            VkSemaphoreCreateInfo semInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };

            if (vkCreateSemaphore(
                m_device.device(),
                &semInfo,
                nullptr,
                &fs.imageAvailable) != VK_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to create Vulkan image-available semaphore.");
            }

            VkFenceCreateInfo fenceInfo{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
            fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

            if (vkCreateFence(
                m_device.device(),
                &fenceInfo,
                nullptr,
                &fs.inFlightFence) != VK_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to create Vulkan frame fence.");
            }
        }

        initSwapchainSync();
    }

    void VulkanBackend::initSwapchainSync()
    {
        destroySwapchainSync();

        const size_t imageCount =
            m_swapchain.images().size();

        m_imageRenderFinished.resize(imageCount, VK_NULL_HANDLE);
        m_imagesInFlight.assign(imageCount, VK_NULL_HANDLE);
        m_swapchainImageLayouts.assign(
            imageCount,
            VK_IMAGE_LAYOUT_UNDEFINED);

        VkSemaphoreCreateInfo semInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };

        for (VkSemaphore& semaphore : m_imageRenderFinished)
        {
            if (vkCreateSemaphore(
                m_device.device(),
                &semInfo,
                nullptr,
                &semaphore) != VK_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to create Vulkan render-finished semaphore.");
            }
        }
    }

    void VulkanBackend::destroyFrameSync()
    {
        VkDevice device = m_device.device();

        if (device == VK_NULL_HANDLE)
            return;

        for (FrameSync& fs : m_frameSync)
        {
            if (fs.imageAvailable != VK_NULL_HANDLE)
            {
                vkDestroySemaphore(
                    device,
                    fs.imageAvailable,
                    nullptr);

                fs.imageAvailable = VK_NULL_HANDLE;
            }

            if (fs.inFlightFence != VK_NULL_HANDLE)
            {
                vkDestroyFence(
                    device,
                    fs.inFlightFence,
                    nullptr);

                fs.inFlightFence = VK_NULL_HANDLE;
            }
        }

        m_frameSync.clear();
    }

    void VulkanBackend::destroySwapchainSync()
    {
        VkDevice device = m_device.device();

        if (device == VK_NULL_HANDLE)
            return;

        for (VkSemaphore semaphore : m_imageRenderFinished)
        {
            if (semaphore != VK_NULL_HANDLE)
            {
                vkDestroySemaphore(
                    device,
                    semaphore,
                    nullptr);
            }
        }

        m_imageRenderFinished.clear();
        m_imagesInFlight.clear();
        m_swapchainImageLayouts.clear();
    }

    void VulkanBackend::onSwapchainRecreated()
    {
        vkDeviceWaitIdle(m_device.device());
        destroySceneResources();
        m_pipelineManager.shutdown();

        m_swapchain.recreate();

        if (!m_swapchain.validForRendering())
            return;

        initSwapchainSync();
        m_pipelineManager.init(m_device.device());
        m_sceneFrameResources.resize(m_frameSync.size());
        m_imageStates.clear();
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

