#include <spdlog/spdlog.h>
#include "ic/renderer/renderer_backend.h"
#include "ic/renderer/frame_graph/compiled_graph_plan.h"
#include "ic/renderer/frame_graph/frame_graph_types.h"

namespace ic
{
    class VulkanBackend : public RendererBackend
    {
    public:

        void initialize(
            const RendererSpecification&) override
        {
            spdlog::info("[VulkanBackend] Initialized");
        }

        void shutdown() override
        {
            spdlog::info("[VulkanBackend] Shutdown");
        }

        void execute(
            const CompiledGraphPlan& plan,
            const FrameContext&) override
        {
            spdlog::info("--- Vulkan Backend Execution ---");

            for (const auto& node : plan.nodes)
            {
                spdlog::info(
                    "Node {} | Queue {} | Type {}",
                    node.nodeId,
                    static_cast<int>(node.queue),
                    static_cast<int>(node.type));
            }

            spdlog::info("------");
        }
    };
}
