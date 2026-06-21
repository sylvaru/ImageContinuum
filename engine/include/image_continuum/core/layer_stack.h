// image_continuum/core/layer_stack.h
#pragma once

#include <vector>
#include <memory>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <utility>
#include <new>
#include <cassert>
#include "image_continuum/interface/layer.h"

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
        void updateAll(float dt);
        void renderAll(float alpha);

    private:
        std::vector<std::unique_ptr<Layer>> m_layers;
    };
}