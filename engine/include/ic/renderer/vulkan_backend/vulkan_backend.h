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
#include "ic/renderer/vulkan_backend/vulkan_graph_resource_registry.h"
#include "ic/renderer/vulkan_backend/vulkan_frame_executor.h"
#include "ic/renderer/vulkan_backend/vulkan_upload_scheduler.h"
#include "ic/renderer/vulkan_backend/vulkan_retirement_queue.h"
#include "ic/renderer/vulkan_backend/vulkan_gpu_scene.h"
#include "ic/renderer/renderer_gpu_assets.h"
#include "ic/renderer/path_tracing/path_tracer_types.h"

#include <glm/glm.hpp>
#include <unordered_map>
#include <array>
#include <vector>

namespace ic
{
    class Window;
    class PipelineLibrary;

    class VulkanBackend final : public RendererBackend
    {
    public:

        void init(
            const RendererSpecification& spec,
            const PipelineLibrary& pipelineLibrary,
            Window& window,
            uint32_t workerCount) override;

        void shutdown() override;

        [[nodiscard]] bool execute(
            const CompiledGraphPlan& plan,
            const GraphExecutionContext& execution,
            const FrameContext& ctx,
            const SceneRenderView& scene) override;

        //SwapchainInfo swapchainInfo() const override;

        std::vector<IBLBakeResult> executeIBLBakeRequests(
            std::span<const IBLBakeRequest> requests,
            const FrameContext& ctx) override;

        bool beginDebugGuiFrame() override;
        void endDebugGuiFrame() override;
        bool vsyncEnabled() const override;
        void setVsyncEnabled(bool enabled) override;
        bool clusteredForwardHeatmapEnabled() const override;
        void setClusteredForwardHeatmapEnabled(bool enabled) override;
        bool hiZDebugViewEnabled() const override;
        void setHiZDebugViewEnabled(bool enabled) override;
        uint32_t hiZDebugMip() const override;
        void setHiZDebugMip(uint32_t mip) override;

        VulkanResourceAllocator& resourceAllocator()
        {
            return m_resourceAllocator;
        }

        const VulkanResourceAllocator& resourceAllocator() const
        {
            return m_resourceAllocator;
        }

        struct ImageState
        {
            VkImageLayout layout;
            AccessType access;
        };

        struct UploadedTexture
        {
            VulkanTexture texture;
            VkImageView view = VK_NULL_HANDLE;
            VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
            uint32_t descriptorIndex = UINT32_MAX;
        };

        struct UploadedSampler
        {
            VkSampler sampler = VK_NULL_HANDLE;
            uint32_t descriptorIndex = UINT32_MAX;
        };

        struct PathTraceResources
        {
            VulkanTexture accumulation;
            VulkanTexture tonemap;
            VkImageView accumulationView = VK_NULL_HANDLE;
            VkImageView tonemapView = VK_NULL_HANDLE;

            std::vector<VulkanBuffer> pathTraceConstants;
            std::vector<VulkanBuffer> tonemapConstants;
            VulkanBuffer sceneVertices;
            VulkanBuffer sceneMaterials;
            VulkanBuffer sceneTriangles;
            VulkanBuffer sceneBvhNodes;

            VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
            std::vector<VkDescriptorSet> pathTraceDescriptorSets;
            std::vector<VkDescriptorSet> tonemapDescriptorSets;
            std::vector<uint64_t> pathTraceDescriptorVersions;
            uint64_t pathTraceDescriptorGeneration = 1;

            VkImageLayout accumulationLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VkImageLayout tonemapLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            uint32_t width = 0;
            uint32_t height = 0;
            uint32_t sceneVertexCount = 0;
            uint32_t sceneMaterialCount = 0;
            uint32_t sceneTriangleCount = 0;
            uint32_t sceneBvhNodeCount = 0;
            uint32_t firstEmissiveTriangleIndex = UINT32_MAX;
            uint32_t accumulatedSampleCount = 0;
            uint64_t sceneVersion = UINT64_MAX;
            uint64_t environmentVersion = UINT64_MAX;
            uint64_t lastSceneBuildFrame = UINT64_MAX;
            bool sceneHadPendingModels = false;
            float tonemapExposure = 1.0f;
            glm::mat4 previousView = glm::mat4(1.0f);
            glm::mat4 previousProjection = glm::mat4(1.0f);
            bool resetAccumulation = true;
            bool hasPreviousCamera = false;
            bool pathTraceDescriptorsDirty = true;
            bool tonemapDescriptorsDirty = true;
        };

        struct EnvironmentResources
        {
            AssetHandle source = {};
            VulkanTexture cubemap;
            VulkanTexture equirect;
            VulkanTexture irradiance;
            VulkanTexture prefiltered;
            VulkanTexture brdfLut;
            VkImageView cubemapView = VK_NULL_HANDLE;
            VkImageView cubemapStorageView = VK_NULL_HANDLE;
            VkImageView equirectView = VK_NULL_HANDLE;
            VkImageView irradianceView = VK_NULL_HANDLE;
            VkImageView irradianceStorageView = VK_NULL_HANDLE;
            VkImageView prefilteredView = VK_NULL_HANDLE;
            std::vector<VkImageView> prefilteredStorageViews;
            VkImageView brdfLutView = VK_NULL_HANDLE;
            VkImageView brdfLutStorageView = VK_NULL_HANDLE;
            VkSampler sampler = VK_NULL_HANDLE;
            VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
            VkDescriptorPool bakeDescriptorPool = VK_NULL_HANDLE;
            VkDescriptorSet convertDescriptorSet = VK_NULL_HANDLE;
            VkDescriptorSet irradianceDescriptorSet = VK_NULL_HANDLE;
            std::vector<VkDescriptorSet> prefilterDescriptorSets;
            VkDescriptorSet brdfLutDescriptorSet = VK_NULL_HANDLE;
            std::vector<VulkanBuffer> skyboxConstants;
            std::vector<VkDescriptorSet> skyboxDescriptorSets;
            VkImageLayout cubemapLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VkImageLayout equirectLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VkImageLayout irradianceLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VkImageLayout prefilteredLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VkImageLayout brdfLutLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            uint32_t cubemapSize = 512;
            uint32_t irradianceSize = 64;
            uint32_t prefilterSize = 256;
            uint32_t prefilterMipCount = 1;
            uint32_t brdfLutSize = 512;
            bool converted = false;
            bool iblBaked = false;
            bool skyboxDescriptorsDirty = true;
        };

        // Clustered-light-grid + Hi-Z debug overlay resources. The GPU-driven
        // cull/indirect-draw buffers live in VulkanGpuScene (m_gpuScene)
        // instead, since they are a separate concern (draw submission, not
        // light clustering or debug visualization).
        struct ClusteredForwardResources
        {
            VulkanBuffer clusterBounds;
            VulkanBuffer clusterLightGrid;
            VulkanBuffer clusterLightIndices;
            VulkanBuffer clusterLightCounter;
            VkSampler hiZDebugSampler = VK_NULL_HANDLE;
            std::vector<VkImageView> hiZDebugViews;
            std::vector<VkDescriptorSet> hiZDebugDescriptors;
            GraphResourceId hiZDebugResource = InvalidGraphResourceId;
            uint64_t hiZDebugGeneration = 0;
            uint32_t width = 0;
            uint32_t height = 0;
            uint32_t hiZMipCount = 0;
            bool loggedHiZ = false;
            bool loggedHiZDebugResource = false;
            uint32_t clusterCountX = 0;
            uint32_t clusterCountY = 0;
            uint32_t clusterCountZ = 0;
            uint32_t clusterCount = 0;
        };

    private:

        void executeGraph(
            const CompiledGraphPlan& plan,
            const GraphExecutionContext& execution,
            const FrameContext& ctx,
            const SceneRenderView& scene,
            VkImage swapchainImage,
            VkImageLayout swapchainInitialLayout,
            std::vector<VkCommandBuffer>& commandBuffers);

        void initImGui(Window& window);
        void shutdownImGui();
        void recordImGui(
            const FrameContext& ctx,
            VkImage swapchainImage,
            std::vector<VkCommandBuffer>& commandBuffers);

        void applyBarriers(
            VkCommandBuffer cmd,
            const CompiledGraphPlan& plan,
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
            VkImageLayout swapchainInitialLayout,
            QueueType sourceQueue,
            QueueType destinationQueue,
            bool crossQueueRelease,
            bool crossQueueAcquire,
            QueueType commandQueue);

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
            const SceneRenderView& scene,
            VkCommandBuffer cmd);

        void executePathTraceNode(
            const CompiledGraphPlan& plan,
            const ExecutionNode& node,
            const FrameContext& ctx,
            const SceneRenderView& scene,
            VkCommandBuffer cmd);

        void executeEnvironmentConvertNode(
            const CompiledGraphPlan& plan,
            const ExecutionNode& node,
            const FrameContext& ctx,
            const SceneRenderView& scene,
            VkCommandBuffer cmd);

        void executeTonemapNode(
            const CompiledGraphPlan& plan,
            const ExecutionNode& node,
            const FrameContext& ctx,
            VkCommandBuffer cmd);

        void executeTransferNode(
            const CompiledGraphPlan& plan,
            const ExecutionNode& node,
            const FrameContext& ctx,
            VkCommandBuffer cmd,
            VkImage swapchainImage);

        void destroySceneResources();
        void ensurePathTraceResources();
        void ensurePathTraceSceneResources(
            const FrameContext& ctx,
            const SceneRenderView& scene);
        void destroyPathTraceResources();
        void retirePathTraceSceneResources();
        void uploadPathTraceScene(const PathTraceSceneData& sceneData);
        VkImageView createTextureView(const VulkanTexture& texture) const;
        void updatePathTraceDescriptors(
            const VulkanComputePipeline* pathTracePipeline,
            const VulkanComputePipeline* tonemapPipeline,
            uint32_t frameSlot);
        bool ensureEnvironmentResources(
            const FrameContext& ctx,
            const SceneRenderView& scene,
            VkCommandBuffer cmd);
        VulkanComputePipeline* environmentConvertPipeline();
        bool convertEnvironmentIfReady(
            VulkanComputePipeline& pipeline,
            const FrameContext& ctx,
            const SceneRenderView& scene,
            VkCommandBuffer cmd);
        void updateSkyboxDescriptors(const VulkanGraphicsPipeline& pipeline);
        void destroyEnvironmentResources();
        void drawSkybox(
            VulkanGraphicsPipeline& pipeline,
            const FrameContext& ctx,
            const SceneRenderView& scene,
            VkCommandBuffer cmd);
        void ensureDepthTarget();
        void destroyDepthTarget();
        void drawHiZDebugWindow();
        void ensureHiZDebugDescriptors(VulkanGraphResourceEntry& hiZ);
        void destroyHiZDebugDescriptors();
        void ensureComputeTestResources(
            const VulkanComputePipeline& pipeline);
        void ensureClusteredForwardResources();
        void ensureGpuDrivenResources();
        bool bindClusteredForwardCompute(
            const VulkanComputePipeline& pipeline,
            const FrameContext& ctx,
            const SceneRenderView& scene,
            VkCommandBuffer cmd);
        void bindClusteredForwardGraphics(
            const VulkanGraphicsPipeline& pipeline,
            const FrameContext& ctx,
            VkCommandBuffer cmd);
        void readbackVisibleInstanceCount(
            const FrameContext& ctx,
            VkCommandBuffer cmd);
        void destroyClusteredForwardResources();

        VulkanUploadedModel* requestModel(
            AssetHandle handle,
            const AssetManager& assets);

        uint32_t requestTexture(
            AssetHandle modelHandle,
            uint32_t imageIndex,
            const ImageAsset& image,
            TextureTransferFunction transfer,
            bool asynchronous = true);

        uint32_t requestSampler(const SamplerAsset* sampler);

        bool prepareSceneResources(
            const FrameContext& ctx,
            const SceneRenderView& scene,
            GraphicsPipelineHandle pipelineHandle,
            bool updateGraphicsDescriptors = true);

        GraphicsPipelineHandle pipelineForNode(
            const CompiledGraphPlan& plan,
            const ExecutionNode& node);

        ComputePipelineHandle computePipelineForNode(
            const CompiledGraphPlan& plan,
            const ExecutionNode& node);

        void updateFrameDescriptors(
            VulkanGpuSceneFrameResources& resources,
            GraphicsPipelineHandle pipelineHandle);

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
            VkPipelineStageFlags dstStage,
            uint32_t baseMipLevel = 0,
            uint32_t levelCount = 1);

        VulkanGraphResourceRegistry m_graphResourceRegistry;
        VulkanFrameExecutor m_frameExecutor;
        VulkanUploadScheduler m_uploadScheduler;
        VulkanRetirementQueue m_retirementQueue;
        std::unordered_map<VkImage, ImageState> m_imageStates;

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
        ClusteredForwardResources m_clusteredForwardResources;
        PathTraceResources m_pathTraceResources;
        EnvironmentResources m_environmentResources;

        const PipelineLibrary* m_pipelineLibrary = nullptr;
        std::unordered_map<PipelineId, GraphicsPipelineHandle, PipelineIdHash> m_pipelineHandles;
        std::unordered_map<PipelineId, ComputePipelineHandle, PipelineIdHash> m_computePipelineHandles;
        std::unordered_map<AssetHandle, VulkanUploadedModel, AssetHandleHash> m_uploadedModels;
        std::unordered_map<uint64_t, UploadedTexture> m_uploadedTextures;
        std::unordered_map<uint64_t, UploadedSampler> m_uploadedSamplers;
        VulkanGpuScene m_gpuScene;
        uint32_t m_workerSlots = 1;
        bool m_imguiEnabled = false;
        bool m_imguiFrameActive = false;
        bool m_clusteredForwardHeatmapEnabled = false;
        bool m_hiZDebugViewEnabled = false;
        uint32_t m_hiZDebugMip = 0;
        std::string m_imguiIniPath;
    };
}
