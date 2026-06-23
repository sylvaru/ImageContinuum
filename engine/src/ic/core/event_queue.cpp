// src/core/event_queue.cpp
#include "ic/common/ic_pch.h"
#include "ic/core/event_queue.h"
#include <spdlog/spdlog.h>

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