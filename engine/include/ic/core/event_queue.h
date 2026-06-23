// ic/core/event_queue.h
#pragma once
#include <concurrentqueue.h>
#include "ic/core/events.h"

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