#pragma once

// 
// C++ STANDARD LIBRARY (High-frequency data structures & memory management)
//
#include <utility>
#include <algorithm>
#include <functional>

// Containers 
// (Contiguous memory structures for Data-Oriented Design)
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

// Concurrency & Synchronization (For job systems and resource queuing)
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>

//
// OPERATING SYSTEM & PLATFORM ABSTRACTION 
//
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

//
// GRAPHICS API & THIRD-PARTY INTEGRATION HOOKS
//

// Vulkan Headers
#define VK_NO_PROTOTYPES
//#include <vulkan/vulkan.h>

// Windowing / Cross-platform input
// #include <GLFW/glfw3.h>

// Math Library (e.g., GLM configured for Vulkan depth conventions)
//#define GLM_FORCE_RADIANS
//#define GLM_FORCE_DEPTH_ZERO_TO_ONE
//#include <glm/glm.hpp>
//#include <glm/gtc/matrix_transform.hpp>
//#include <glm/gtc/type_ptr.hpp>

// 
// ENGINE LOGGING / UTILITY SHORTCUTS (debugging helpers)
// 

// Central logging or assert statements exist globally within this PCH 
// ensures they are immediately accessible across every subsystem.

namespace ic {
    // Basic inline logging placeholder for immediate feedback
    inline void logInfo(std::string_view message) {
        std::cout << "[IC_INFO] " << message << '\n';
    }

    inline void logError(std::string_view message) {
        std::cerr << "[IC_ERROR] " << message << '\n';
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