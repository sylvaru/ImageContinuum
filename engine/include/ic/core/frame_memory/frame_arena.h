#pragma once
#include "frame_arena_region.h"
#include "frame_arena_storage.h"
#include "frame_memory_types.h"

namespace ic
{

	class FrameArena
	{
	public:
		explicit FrameArena(size_t size);
		FrameArena(const FrameArena&) = delete;
		FrameArena& operator=(const FrameArena&) = delete;

		void reset();

		FrameArenaRegion& region(FrameRegion r)
		{
			return m_regions[(size_t)r];
		}

		template<typename T>
		T* alloc(FrameRegion r, size_t count = 1)
		{
			return static_cast<T*>(
				m_regions[(size_t)r].allocate(
					sizeof(T) * count,
					alignof(T)));
		}

	private:
		FrameArenaStorage m_storage;
		std::array<FrameArenaRegion, (size_t)FrameRegion::Count> m_regions;
	};
}