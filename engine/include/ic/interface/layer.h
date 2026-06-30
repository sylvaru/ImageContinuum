// ic/interface/layer.h
#pragma once

namespace ic
{
	struct Event;
    struct AppServices;
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

        virtual void onAttach(AppServices& services) { (void)services; }
        virtual void onAttack() {}
        virtual void onDetach() {}
        virtual void onUpdate(FrameContext& ctx) = 0;
        virtual void onRender(float alpha) = 0;
        virtual bool onEvent(const Event&) { return false; } // Only for system/UI events. false = not consumed

        const LayerSpecification& spec() const { return m_spec;  }

    private:
        LayerSpecification m_spec;
    };
}
