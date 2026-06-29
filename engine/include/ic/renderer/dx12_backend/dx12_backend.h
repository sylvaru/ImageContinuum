#pragma once
#include "ic/renderer/renderer_backend.h"
#include "ic/renderer/frame_graph/compiled_graph_plan.h"
#include "ic/renderer/frame_graph/frame_graph_types.h"

#include <spdlog/spdlog.h>
#include <d3d12.h>


namespace ic
{
    class Window;

	class DX12Backend : public RendererBackend
	{
	public:
        void initialize(
            [[maybe_unused]] const RendererSpecification& spec,
            [[maybe_unused]] Window& window,
            [[maybe_unused]] uint32_t workerCount) override
        {
            spdlog::info("[DX12Backend] Initialized");
        }

        void shutdown() override
        {
            spdlog::info("[DX12Backend] Shutdown");
        }

        void execute(
            [[maybe_unused]] const CompiledGraphPlan& plan,
            [[maybe_unused]] const FrameContext& ctx) override
        {

        }

    private:

        
	};
}