// ic/core/memory/frame_arena_manager.h
#pragma once
#include "frame_arena.h"
#include <array>
#include <utility>

namespace ic
{
	template<size_t FrameCount>
	class FrameArenaManager
	{
	public:

		explicit FrameArenaManager(size_t arenaSize)
			: m_arenas(createArenas(arenaSize, std::make_index_sequence<FrameCount>{}))
		{
		}

		FrameArena& beginFrame(uint64_t frameIndex)
		{
			return m_arenas[frameIndex % FrameCount];
		}

	private:

		template<size_t... Is>
		static std::array<FrameArena, FrameCount> 
			createArenas(size_t arenaSize, std::index_sequence<Is...>)
		{
			return { ((void)Is, FrameArena(arenaSize))... };
		}

		std::array<
			FrameArena,
			FrameCount> m_arenas;
	};
}