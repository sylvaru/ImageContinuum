// ic/interface/layer.h
#pragma once

namespace ic
{
	struct Event;
    struct FrameContext;
    struct LayerSpecification
    {
        std::string debugName = "Layer";
    };

    class Layer
    {
    public:
        Layer(const LayerSpecification& spec = {}) : m_spec(spec) {}
        virtual ~Layer() = default;

        virtual void onUpdate(FrameContext& ctx) = 0;
        virtual void onRender(float alpha) = 0;
        virtual void onEvent(const Event&) {} // Only for system/UI events

        const LayerSpecification& spec() const { return m_spec;  }

    private:
        LayerSpecification m_spec;
    };
}