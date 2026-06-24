#pragma once
#include <cstdint>

namespace ic 
{

    struct TextureHandle
    {
        uint32_t value = Invalid;

        static constexpr uint32_t Invalid = 0xFFFFFFFFu;

        uint32_t index() const
        {
            return value & 0xFFFFFu;
        }

        uint32_t generation() const
        {
            return value >> 20;
        }
    };

    struct BufferHandle
    {
        uint32_t value = Invalid;

        static constexpr uint32_t Invalid = 0xFFFFFFFFu;

        uint32_t index() const
        {
            return value & 0xFFFFFu;
        }

        uint32_t generation() const
        {
            return value >> 20;
        }
    };

    struct PipelineHandle
    {
        uint32_t value = Invalid;

        static constexpr uint32_t Invalid = 0xFFFFFFFFu;

        uint32_t index() const
        {
            return value & 0xFFFFFu;
        }

        uint32_t generation() const
        {
            return value >> 20;
        }
    };

    struct MaterialHandle
    {
        uint32_t value = Invalid;

        static constexpr uint32_t Invalid = 0xFFFFFFFFu;

        uint32_t index() const
        {
            return value & 0xFFFFFu;
        }

        uint32_t generation() const
        {
            return value >> 20;
        }
    };

    struct MeshHandle
    {
        uint32_t value = Invalid;

        static constexpr uint32_t Invalid = 0xFFFFFFFFu;

        uint32_t index() const
        {
            return value & 0xFFFFFu;
        }

        uint32_t generation() const
        {
            return value >> 20;
        }
    };

}