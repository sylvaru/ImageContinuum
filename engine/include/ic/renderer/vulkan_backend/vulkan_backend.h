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
            [[maybe_unused]] const CompiledGraphPlan& plan,
            [[maybe_unused]] const FrameContext& ctx) override
        {
        }
    };
}
