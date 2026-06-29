#include "ic/renderer/renderer_backend.h"
#include "ic/renderer/frame_graph/compiled_graph_plan.h"
#include "ic/renderer/frame_graph/frame_graph_types.h"
#include "ic/renderer/vulkan_backend/vulkan_instance.h"
#include "ic/renderer/vulkan_backend/vulkan_platform.h"
#include "ic/renderer/vulkan_backend/vulkan_adapter.h"
#include "ic/renderer/vulkan_backend/vulkan_device.h"
#include "ic/renderer/vulkan_backend/vulkan_swapchain.h"
#include "ic/renderer/vulkan_backend/vulkan_command_system.h"

namespace ic
{
    class Window;

    class VulkanBackend : public RendererBackend
    {
    public:

        void initialize(
            const RendererSpecification& spec,
            Window& window,
            uint32_t workerCount) override;

        void shutdown() override;

        void execute(
            const CompiledGraphPlan& plan,
            const FrameContext& ctx) override;

        struct FrameSync
        {
            VkFence inFlightFence = VK_NULL_HANDLE;
            VkSemaphore imageAvailable = VK_NULL_HANDLE;
        };

        struct VulkanResource
        {
            VkImage image;
            VkBuffer buffer;
            VkImageLayout currentLayout;
        };

        struct ImageState
        {
            VkImageLayout layout;
            AccessType access;
        };

    private:

        void executeGraph(
            const CompiledGraphPlan& plan,
            const FrameContext& ctx,
            VkCommandBuffer cmd,
            VkImage swapchainImage);

        void applyBarriers(
            VkCommandBuffer cmd,
            std::span<const ResourceBarrier> barriers,
            std::span<const GraphResource> resources,
            const ExecutionNode& node,
            VkImage swapchainImage);

        void dispatchNode(
            const ExecutionNode& node,
            const FrameContext& ctx,
            VkCommandBuffer cmd,
            VkImage swapchainImage);

        void recordBarrier(
            VkCommandBuffer cmd,
            const ResourceBarrier& barrier,
            std::span<const GraphResource> resources,
            VkImage swapchainImage);

        void submitFrame(
            VkCommandBuffer cmd,
            FrameSync& sync,
            VkSemaphore renderFinished);

        void executeGraphicsNode(
            const ExecutionNode& node,
            const FrameContext& ctx,
            VkCommandBuffer cmd,
            VkImage swapchainImage);

        void initFrameSync(const RendererSpecification& spec);
        void initSwapchainSync();
        void destroyFrameSync();
        void destroySwapchainSync();
        void onSwapchainRecreated();

        VkImageLayout usageToLayout(ResourceUsage usage) const;

        VkAccessFlags toAccessMask(AccessType type) const;
        VkAccessFlags accessMaskFor(
            ResourceUsage usage,
            AccessType access) const;
        VkPipelineStageFlags pipelineStageFor(
            ResourceUsage usage,
            AccessType access) const;

        VkImageLayout getOrInitImageLayout(VkImage image);

        void transitionImage(
            VkCommandBuffer cmd,
            VkImage image,
            VkImageLayout oldLayout,
            VkImageLayout newLayout,
            VkAccessFlags srcAccess,
            VkAccessFlags dstAccess,
            VkPipelineStageFlags srcStage,
            VkPipelineStageFlags dstStage);

        std::unordered_map<GraphResourceId, VulkanResource> m_resources;
        std::unordered_map<VkImage, ImageState> m_imageStates;

        std::vector<FrameSync> m_frameSync;
        uint32_t m_currentSwapchainImage = 0;
        std::vector<VkSemaphore> m_imageRenderFinished;
        std::vector<VkFence> m_imagesInFlight;

        VulkanInstance m_instance;
        VulkanPlatform m_platform;
        VulkanAdapter m_adapter;
        VulkanDevice m_device;
        VulkanSwapchain m_swapchain;
        VulkanCommandSystem m_commandSystem;
    };
}
