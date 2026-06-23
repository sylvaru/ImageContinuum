#pragma once
#include <cstdint>

namespace ic 
{

    // 32 bit Generational Handles. 20 bits for indexing (1M distinct objects), 
    // 12 bits for generational tracking to eliminate dangling handle bugs.
    struct TextureHandle
    {
        uint32_t index : 20;
        uint32_t generation : 12;

        inline bool isValid() const { return index != 0xFFFFF; }
    };

    struct BufferHandle
    {
        uint32_t index : 20;
        uint32_t generation : 12;

        inline bool isValid() const { return index != 0xFFFFF; }
    };

    struct PipelineHandle
    {
        uint32_t index : 20;
        uint32_t generation : 12;

        inline bool isValid() const { return index != 0xFFFFF; }
    };

}