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

// Math Library 
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>