// ic/core/layer_stack.h
#pragma once

#include <vector>
#include <memory>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <utility>
#include <new>
#include <cassert>
#include "ic/interface/layer.h"

namespace ic 
{
    struct Event;

    class LayerStack
    {
    public:
        LayerStack() = default;
        ~LayerStack();

        template<typename T, typename... Args>
        T& emplaceLayer(Args&&... args)
        {
            auto layer =
                std::make_unique<T>(
                    std::forward<Args>(args)...);

            T& ref = *layer;

            m_layers.push_back(std::move(layer));

            return ref;
        }
        void dispatchEvent(Event& e);
        void updateAll(FrameContext& ctx);
        void renderAll(float alpha);

    private:
        std::vector<std::unique_ptr<Layer>> m_layers;
    };
}