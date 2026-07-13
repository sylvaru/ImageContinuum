// ic/core/memory/frame_arena.h
#include "ic/common/ic_pch.h"
#include "ic/core/frame_memory/frame_arena.h"

namespace ic
{
	FrameArena::FrameArena(size_t size)
		: m_storage(size)
	{
		std::byte* base = static_cast<std::byte*>(m_storage.base());

		size_t perRegion = size / (size_t)FrameRegion::Count;

		for (size_t i = 0; i < (size_t)FrameRegion::Count; i++)
		{
			m_regions[i] = FrameArenaRegion(
				base + (i * perRegion),
				perRegion);
		}
	}
	void FrameArena::reset()
	{
		for (auto& r : m_regions)
			r.reset();
	}

}
