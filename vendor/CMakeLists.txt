cmake_minimum_required(VERSION 3.20)
project(ImageContinuumVendors CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# =========================================================
# Vulkan Headers
# =========================================================
add_library(vulkan-headers INTERFACE)
target_include_directories(vulkan-headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/vulkan-headers/include>
)

# =========================================================
# Toml++
# =========================================================
set(TOMLPP_BUILD_SINGLE_HEADER OFF CACHE BOOL "" FORCE)
set(TOMLPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(tomlplusplus EXCLUDE_FROM_ALL)

# =========================================================
# GLFW
# =========================================================
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE) # reduces Linux clutter
add_subdirectory(glfw EXCLUDE_FROM_ALL)

# =========================================================
# Spdlog
# =========================================================
set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(SPDLOG_FMT_EXTERNAL OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
add_subdirectory(spdlog EXCLUDE_FROM_ALL)

# =========================================================
# Miniaudio
# =========================================================
add_library(miniaudio INTERFACE)
target_include_directories(miniaudio INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/miniaudio>
)

# =========================================================
# EnTT
# =========================================================
set(ENTT_BUILD_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(entt EXCLUDE_FROM_ALL)

# =========================================================
# GLM
# =========================================================
add_library(glm INTERFACE)
target_include_directories(glm INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/glm>
)

# =========================================================
# ConcurrentQueue
# =========================================================
add_library(concurrentqueue INTERFACE)
target_include_directories(concurrentqueue INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/concurrentqueue>
)

# =========================================================
# FastGltf
# =========================================================
set(FASTGLTF_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(FASTGLTF_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(fastgltf EXCLUDE_FROM_ALL)

# =========================================================
# STB
# =========================================================
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/stb>
)

# =========================================================
# Tracy
# =========================================================
set(TRACY_ENABLE OFF CACHE BOOL "" FORCE)
set(TRACY_ON_DEMAND OFF CACHE BOOL "" FORCE)
set(TRACY_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TRACY_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
add_subdirectory(tracy EXCLUDE_FROM_ALL)


# =========================================================
# ImGui
# =========================================================
set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/imgui")

add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
)

target_include_directories(imgui PUBLIC
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)

target_link_libraries(imgui PUBLIC
    glfw
    vulkan-headers
)


# =========================================================
# Final aggregated interface target
# =========================================================
add_library(ic_thirdparty INTERFACE)

add_dependencies(ic_thirdparty concurrentqueue)

target_link_libraries(ic_thirdparty INTERFACE
    vulkan-headers
    glfw
    spdlog
    miniaudio
    glm
    concurrentqueue
    stb
    imgui
    tomlplusplus::tomlplusplus
    fastgltf::fastgltf
    EnTT::EnTT
    Tracy::TracyClient
)