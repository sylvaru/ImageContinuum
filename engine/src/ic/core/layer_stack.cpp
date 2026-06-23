// ic/core/layer_stack.cpp
#include "ic/common/ic_pch.h"
#include "ic/core/layer_stack.h"
#include "ic/core/events.h"

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

	void LayerStack::updateAll(FrameContext& ctx)
	{
		for (auto& layer : m_layers)
		{
			layer->onUpdate(ctx);
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