// engine/include/image_continuum/core/layer_stack.h
#pragma once

#include <vector>
#include <memory>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <utility>
#include <new>
#include <cassert>

namespace ic {

    // Concept enforcing what a Layer must look like (compile time duck typing)
    template <typename T>
    concept CrtpLayer = requires(T layer, float dt, float alpha) {
        { layer.onUpdate(dt) } -> std::same_as<void>;
        { layer.onRender(alpha) } -> std::same_as<void>;
    };

    class LayerStack
    {
    private:

        // Compile time constant for the Small Buffer Optimization size
        // Adjust this depending on the largest layer you expect to push
        static constexpr size_t INLINE_STORAGE_SIZE = 128;

        // Function pointer types matching the exact signature of the dispatch logic
        using UpdateFunc = void(*)(void*, float);
        using RenderFunc = void(*)(void*, float);
        using DtorFunc   = void(*)(void*);

       
        // Storage node that manages inline memory and raw dispatch function pointers
        // Zero v-tables, zero inheritance, zero overrides
        struct Node {
            alignas(std::max_align_t) std::byte storage[INLINE_STORAGE_SIZE];

            UpdateFunc update_fn = nullptr;
            RenderFunc render_fn = nullptr;
            DtorFunc   dtor_fn = nullptr;

            Node() = default;

            template <CrtpLayer T>
            explicit Node(T layer) {
                static_assert(sizeof(T) <= INLINE_STORAGE_SIZE,
                    "Layer implementation exceeds INLINE_STORAGE_SIZE buffer allocation.");

                // Placement-new the concrete CRTP layer directly into the byte buffer
                T* layer_ptr = ::new (static_cast<void*>(storage)) T(std::move(layer));
                (void)layer_ptr;

                // Assign highly optimized, type erased function pointers
                update_fn = [](void* storage_ptr, float dt) {
                    static_cast<T*>(storage_ptr)->onUpdate(dt);
                    };

                render_fn = [](void* storage_ptr, float alpha) {
                    static_cast<T*>(storage_ptr)->onRender(alpha);
                    };

                dtor_fn = [](void* storage_ptr) {
                    static_cast<T*>(storage_ptr)->~T();
                    };
            }

            ~Node() {
                if (dtor_fn) {
                    dtor_fn(storage);
                }
            }

            // Rule of 5 / Rule of 0 adjustments for raw memory buffers (Move-only)
            Node(const Node&) = delete;
            Node& operator=(const Node&) = delete;

            Node(Node&& other) noexcept {
                std::memcpy(storage, other.storage, INLINE_STORAGE_SIZE);
                update_fn = other.update_fn;
                render_fn = other.render_fn;
                dtor_fn = other.dtor_fn;

                // Nullify the moved-from node so its destructor doesn't double-destroy the memory
                other.dtor_fn = nullptr;
            }

            Node& operator=(Node&& other) noexcept {
                if (this != &other) {
                    if (dtor_fn) {
                        dtor_fn(storage);
                    }
                    std::memcpy(storage, other.storage, INLINE_STORAGE_SIZE);
                    update_fn = other.update_fn;
                    render_fn = other.render_fn;
                    dtor_fn = other.dtor_fn;

                    other.dtor_fn = nullptr;
                }
                return *this;
            }
        };

    public:
        LayerStack() = default;
        ~LayerStack() = default;

        LayerStack(const LayerStack&) = delete;
        LayerStack& operator=(const LayerStack&) = delete;

        LayerStack(LayerStack&&) noexcept = default;
        LayerStack& operator=(LayerStack&&) noexcept = default;

        // Push layer by value or rvalue, allocating it into a uniform pointer array
        template <typename T>
            requires CrtpLayer<T>
        void pushLayer(T layer) {
            // Under the hood, std::make_unique will allocate one contiguous block 
            // for the Model structure, keeping the Layer data tightly bound to the vptr
            m_nodes.emplace_back(std::move(layer));
        }

        // Combine updates and renders into a single contiguous iteration pass
        // to maximize CPU cache line residency
        void updateAndRenderAll(float dt, float alpha) {
            // Iterating once over the array means the CPU pulls the data into L1/L2 cache once, 
            // rather than iterating twice over disjoint vectors.
            for (auto& node : m_nodes) {
                assert(node.update_fn && node.render_fn);
                node.update_fn(node.storage, dt);
                node.render_fn(node.storage, alpha);
            }
        }

    private:
        // Contiguous array of inline nodes. This guarantees that layer instances 
        // sit sequentially in the CPU cache lines
        std::vector<Node> m_nodes;
    };
}