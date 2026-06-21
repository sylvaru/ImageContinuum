// image_continuum/core/event_bus.cpp
#include "image_continuum/core/event_bus.h"

namespace ic
{
    void EventBus::push(
        EventChannel channel,
        const Event& event)
    {
        switch (channel)
        {
        case EventChannel::Input:
            m_input.push(event);
            break;

        case EventChannel::Window:
            m_window.push(event);
            break;

        case EventChannel::System:
            m_system.push(event);
            break;

        case EventChannel::Render:
            m_render.push(event);
            break;
        }
    }
}