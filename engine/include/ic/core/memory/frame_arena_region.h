// ic/core/memory/frame_arena_region.h
#pragma once
#include <cstddef>
#include <array>
#include <cassert>

namespace ic
{

    class FrameArenaRegion
    {
    public:
        FrameArenaRegion() = default;

        FrameArenaRegion(std::byte* base, size_t capacity)
            : m_base(base)
            , m_capacity(capacity)
        {
        }

        void reset()
        {
            m_offset = 0;
            m_peak = 0;
        }

        void* allocate(size_t size, size_t alignment)
        {
            size_t aligned = (m_offset + alignment - 1) & ~(alignment - 1);

            if (aligned + size > m_capacity)
            {
                assert(false && "FrameArenaRegion overflow");
                return nullptr;
            }

            void* ptr = m_base + aligned;
            m_offset = aligned + size;

            if (m_offset > m_peak)
                m_peak = m_offset;

            return ptr;
        }

        size_t offset() const { return m_offset; }
        size_t capacity() const { return m_capacity; }
        size_t peak() const { return m_peak; }

    private:
        std::byte* m_base = nullptr;
        size_t m_capacity = 0;

        size_t m_offset = 0;
        size_t m_peak = 0;
    };
}