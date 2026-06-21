// image_continuum/core/eventbus.h
#pragma once
#include "event_queue.h"

namespace ic
{
	struct Event;


	class EventBus
	{
	public:
		void push(EventChannel channel, const Event& event);

        template<typename Fn>
        void drain(
            EventChannel channel,
            Fn&& fn)
        {
            switch (channel)
            {
            case EventChannel::Input:
                m_input.drain(std::forward<Fn>(fn));
                break;

            case EventChannel::Window:
                m_window.drain(std::forward<Fn>(fn));
                break;

            case EventChannel::System:
                m_system.drain(std::forward<Fn>(fn));
                break;

            case EventChannel::Render:
                m_render.drain(std::forward<Fn>(fn));
                break;
            }
        }

	private:
		EventQueue m_input;
		EventQueue m_window;
		EventQueue m_system;
		EventQueue m_render;
	};
}