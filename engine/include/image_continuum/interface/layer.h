// image_continuum/interface/layer.h
#pragma once

namespace ic
{
	struct Event;

    struct Layer
    {
    public:
        virtual ~Layer() = default;

        virtual void onUpdate(float dt) = 0;
        virtual void onRender(float alpha) = 0;
        virtual void onEvent(Event&) = 0;
    };
}