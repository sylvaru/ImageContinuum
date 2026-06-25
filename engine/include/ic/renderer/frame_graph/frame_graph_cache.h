#pragma once

namespace ic
{
    struct CompiledFramePlan;
    enum class RenderPath;

    struct FrameGraphKey
    {
        RenderPath renderPath;

        uint32_t width;
        uint32_t height;
    };

    class FrameGraphCache
    {
    public:

        const CompiledFramePlan&
            getOrCompile(
                const FrameGraphKey& key);
    };
}