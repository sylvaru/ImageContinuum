// ic/renderer/renderer_arena.h
#pragma once
#include <memory_resource>
#include <utility>
#include <new>
#include <cstddef>

namespace ic
{
    // For now this is persistent frame graph storage
    class FrameGraphArena
    {
    public:
        explicit FrameGraphArena(
            std::size_t initialBlockSize = 1024 * 1024)
            : m_resource(
                initialBlockSize,
                std::pmr::new_delete_resource())
        {}

        FrameGraphArena(
            const FrameGraphArena&) = delete;

        FrameGraphArena& operator=(
            const FrameGraphArena&) = delete;

        FrameGraphArena(
            FrameGraphArena&&) = delete;


        FrameGraphArena& operator=(
            FrameGraphArena&&) = delete;

        ~FrameGraphArena() = default;

        [[nodiscard]]
        std::pmr::memory_resource*
            resource() noexcept
        {
            return &m_resource;
        }

        template<typename T, typename... Args>
        [[nodiscard]]
        T* create(Args&&... args)
        {
            void* memory =
                m_resource.allocate(
                    sizeof(T),
                    alignof(T));
            return ::new (memory)
                T(std::forward<Args>(args)...);
        }
        // This should only be called during a resize or 
        // top level settings change
        // when the graph is rebuilt and re compiled
        void reset() noexcept {
            m_resource.release();
        }
    private:
        std::pmr::monotonic_buffer_resource m_resource;
    };
}