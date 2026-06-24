#pragma once

namespace ic
{
    class FrameGraphBuilder;

    template<typename T>
    concept FrameGraphPass = requires(
        T pass,
        FrameGraphBuilder & builder)
    {
        { pass.setup(builder) } -> std::same_as<void>;
    };
}