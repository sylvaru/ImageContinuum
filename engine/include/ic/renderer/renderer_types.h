#pragma once

namespace ic
{
    struct RendererMemory
    {
        std::pmr::memory_resource* graph;
        std::pmr::memory_resource* compile;
    };
}