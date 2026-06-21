// image_continuum/core/event_queue.h
#pragma once
#include <cstdint>
#include <vector>
#include <functional>
#include <memory>
#include <concurrentqueue.h>
#include "image_continuum/interface/events.h"

namespace ic
{
    class EventQueue
    {
    public:

		void push(const Event& e);

		bool tryPop(Event& out);

		template<typename Fn>
		void drain(Fn&& fn)
		{
			Event e;

			while (m_queue.try_dequeue(e))
			{
				fn(e);
			}
		}
    private:
        moodycamel::ConcurrentQueue<Event> m_queue;
    };

	struct StateTransitionCommand
	{
		enum class Type 
		{ 
			PushLayer,
			PopLayer,
			ChangeScene 
		};
		uint64_t targetStateID;
	};


}