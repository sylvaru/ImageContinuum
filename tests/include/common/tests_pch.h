#pragma once

// ============================================================================
// C++ STANDARD LIBRARY
// ============================================================================
// Clients need high-frequency utilities, memory management, and basic strings.
#include <utility>
#include <algorithm>
#include <functional>

// Containers (Contiguous memory structures for game logic / data-oriented design)
#include <vector>
#include <array>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>

// Streams & Diagnostics
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <optional>
#include <variant>
#include <memory>

// Concurrency & Synchronization (For client-side jobs or threading)
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>

// External
#include <spdlog/spdlog.h>