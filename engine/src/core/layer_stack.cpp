// core/layer_stack.cpp
#include "image_continuum/common/ic_pch.h"
#include "image_continuum/core/layer_stack.h"
#include "image_continuum/interface/events.h"

namespace ic
{

	void LayerStack::dispatchEvent(Event& e)
	{
		for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it)
		{
			(*it)->onEvent(e);

		}
	}

	LayerStack::~LayerStack() = default;

	void LayerStack::updateAll(float dt)
	{
		for (auto& layer : m_layers)
		{
			layer->onUpdate(dt);
		}
	}

	void LayerStack::renderAll(float alpha)
	{
		for (auto& layer : m_layers)
		{
			layer->onRender(alpha);
		}
	}
}