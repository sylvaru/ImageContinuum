#include "ic/renderer/renderer_backend.h"
#include "ic/renderer/frame_graph/compiled_graph_plan.h"
#include "ic/renderer/frame_graph/frame_graph_types.h"
#include "ic/renderer/vulkan_backend/vulkan_instance.h"
#include "ic/renderer/vulkan_backend/vulkan_platform.h"
#include "ic/renderer/vulkan_backend/vulkan_adapter.h"
#include "ic/renderer/vulkan_backend/vulkan_device.h"
#include "ic/renderer/vulkan_backend/vulkan_swapchain.h"
#include "ic/renderer/vulkan_backend/vulkan_command_system.h"
#include "ic/renderer/vulkan_backend/vulkan_descriptor_system.h"
#include "ic/renderer/vulkan_backend/vulkan_pipeline_manager.h"
#include "ic/renderer/vulkan_backend/vulkan_resource_allocator.h"
#include "ic/renderer/renderer_gpu_assets.h"

#include <glm/glm.hpp>

namespace ic
{
    class Window;
    class PipelineLibrary;

    class VulkanBackend final : public RendererBackend
    {
    public:

        void initialize(
            const RendererSpecification& spec,
            const PipelineLibrary& pipelineLibrary,
            Window& window,
            uint32_t workerCount) override;

        void shutdown() override;

        void execute(
            const CompiledGraphPlan& plan,
            const FrameContext& ctx,
            const SceneRenderView& scene) override;

        VulkanResourceAllocator& resourceAllocator()
        {
            return m_resourceAllocator;
        }

        const VulkanResourceAllocator& resourceAllocator() const
        {
            return m_resourceAllocator;
        }

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

        struct UploadedModel
        {
            VulkanBuffer vertexBuffer;
            VulkanBuffer indexBuffer;
            std::vector<GpuMesh> meshes;
            std::vector<glm::mat4> meshTransforms;
            std::vector<GpuMaterialData> materials;
            bool uploaded = false;
        };

        struct FrameSceneResources
        {
            VulkanBuffer frameConstants;
            VulkanBuffer objects;
            VulkanBuffer materials;

            VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
            VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

            uint32_t objectCapacity = 0;
            uint32_t materialCapacity = 0;
        };

        struct DrawItem
        {
            UploadedModel* model = nullptr;
            uint32_t objectIndex = 0;
            uint32_t meshIndex = 0;
            uint32_t materialIndex = 0;
        };

    private:

        void executeGraph(
            const CompiledGraphPlan& plan,
            const FrameContext& ctx,
            const SceneRenderView& scene,
            VkImage swapchainImage,
            VkImageLayout swapchainInitialLayout,
            std::vector<VkCommandBuffer>& commandBuffers);

        void applyBarriers(
            VkCommandBuffer cmd,
            std::span<const ResourceBarrier> barriers,
            std::span<const GraphResource> resources,
            const ExecutionNode& node,
            VkImage swapchainImage,
            VkImageLayout swapchainInitialLayout);

        void dispatchNode(
            const CompiledGraphPlan& plan,
            const ExecutionNode& node,
            const FrameContext& ctx,
            const SceneRenderView& scene,
            VkCommandBuffer cmd,
            VkImage swapchainImage);

        void recordBarrier(
            VkCommandBuffer cmd,
            const ResourceBarrier& barrier,
            std::span<const GraphResource> resources,
            VkImage swapchainImage,
            VkImageLayout swapchainInitialLayout);

        void submitFrame(
            std::span<const VkCommandBuffer> commandBuffers,
            FrameSync& sync,
            VkSemaphore renderFinished);

        void executeGraphicsNode(
            const CompiledGraphPlan& plan,
            const ExecutionNode& node,
            const FrameContext& ctx,
            const SceneRenderView& scene,
            VkCommandBuffer cmd,
            VkImage swapchainImage);

        void executeComputeNode(
            const CompiledGraphPlan& plan,
            const ExecutionNode& node,
            const FrameContext& ctx,
            VkCommandBuffer cmd);

        void executeTransferNode(
            const CompiledGraphPlan& plan,
            const ExecutionNode& node,
            const FrameContext& ctx,
            VkCommandBuffer cmd);

        void destroySceneResources();
        void ensureDepthTarget();
        void destroyDepthTarget();
        void ensureComputeTestResources(
            const VulkanComputePipeline& pipeline);

        UploadedModel* requestModel(
            AssetHandle handle,
            const AssetManager& assets);

        bool prepareSceneResources(
            const FrameContext& ctx,
            const SceneRenderView& scene,
            GraphicsPipelineHandle pipelineHandle,
            std::vector<DrawItem>& draws);

        GraphicsPipelineHandle pipelineForNode(
            const CompiledGraphPlan& plan,
            const ExecutionNode& node);

        ComputePipelineHandle computePipelineForNode(
            const CompiledGraphPlan& plan,
            const ExecutionNode& node);

        void updateFrameDescriptors(
            FrameSceneResources& resources,
            GraphicsPipelineHandle pipelineHandle);

        void initFrameSync(const RendererSpecification& spec);
        void initSwapchainSync();
        void destroyFrameSync();
        void destroySwapchainSync();
        void onSwapchainRecreated();
        TextureFormat swapchainTextureFormat() const;

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
        std::vector<VkImageLayout> m_swapchainImageLayouts;

        VulkanInstance m_instance;
        VulkanPlatform m_platform;
        VulkanAdapter m_adapter;
        VulkanDevice m_device;
        VulkanSwapchain m_swapchain;
        VulkanCommandSystem m_commandSystem;
        VulkanDescriptorSystem m_descriptorSystem;
        VulkanPipelineManager m_pipelineManager;
        VulkanResourceAllocator m_resourceAllocator;

        VulkanTexture m_depthTexture;
        VkImageView m_depthImageView = VK_NULL_HANDLE;
        VkImageLayout m_depthLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        uint32_t m_depthWidth = 0;
        uint32_t m_depthHeight = 0;
        VulkanBuffer m_computeTestBuffer;
        VkDescriptorPool m_computeTestDescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_computeTestDescriptorSet = VK_NULL_HANDLE;

        const PipelineLibrary* m_pipelineLibrary = nullptr;
        std::unordered_map<PipelineId, GraphicsPipelineHandle, PipelineIdHash> m_pipelineHandles;
        std::unordered_map<PipelineId, ComputePipelineHandle, PipelineIdHash> m_computePipelineHandles;
        std::unordered_map<AssetHandle, UploadedModel, AssetHandleHash> m_uploadedModels;
        std::vector<FrameSceneResources> m_sceneFrameResources;
        uint32_t m_workerSlots = 1;
    };
}
