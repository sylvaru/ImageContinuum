#pragma once

// C++ STANDARD LIBRARY (High-frequency data structures & memory management)
#include <utility>
#include <algorithm>
#include <functional>

// Containers (Contiguous memory structures for Data-Oriented Design)
#include <vector>
#include <array>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>

// Streams & Diagnostics (Keep I/O minimal in production render paths)
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <optional>
#include <variant>
#include <memory>

// Concurrency & Synchronization (For job systems and resource queuing)
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>

// OPERATING SYSTEM & PLATFORM ABSTRACTION 
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

// GRAPHICS API & THIRD-PARTY INTEGRATION HOOKS
// Vulkan Headers
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

// Windowing / Cross-platform input
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

// Math Library (Configured for Vulkan depth conventions)
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Logging & Entity Component System
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

#include <entt/entt.hpp>

// ENGINE LOGGING / UTILITY SHORTCUTS (debugging helpers)
namespace ic {
    // Basic inline logging placeholder for immediate feedback
    inline void logInfo(std::string_view message) {
        spdlog::info("[IC_INFO] {}", message);
    }

    inline void logError(std::string_view message) {
        spdlog::error("[IC_ERROR] {}", message);
    }
}

// Global Assert Macro for debugging/development configurations
#ifndef NDEBUG
#define IC_ASSERT(x, msg) do { \
        if (!(x)) { \
            std::cerr << "Assertion Failed: " << msg << " at " << __FILE__ << ":" << __LINE__ << '\n'; \
            std::abort(); \
        } \
    } while(0)
#else
#define IC_ASSERT(x, msg) do {} while(0)
#endif