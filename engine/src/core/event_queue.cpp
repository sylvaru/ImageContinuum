// image_continuum/core/event_queue.cpp
#include "image_continuum/common/ic_pch.h"
#include "image_continuum/core/event_queue.h"


namespace ic
{
	void EventQueue::push(const Event& e)
	{
		m_queue.enqueue(e);
	}

	bool EventQueue::tryPop(Event& out)
	{
		return m_queue.try_dequeue(out);
	}
}