# engine/CMakeLists.txt

# Determine the library type
if(IC_BUILD_SHARED)
    set(LIB_TYPE SHARED)
else()
    set(LIB_TYPE STATIC)
endif()

# Define the Engine Target
add_library(ImageContinuumEngine ${LIB_TYPE})

# Add source files
target_sources(ImageContinuumEngine PRIVATE
    "include/ic/core/app_base.h" 
    "include/ic/core/event_queue.h" 
    "include/ic/core/event_frame_buffer.h" 
    "include/ic/core/frame.h"
    "include/ic/core/frame_executor.h"

    "include/ic/common/ic_common.h" 
    "include/ic/common/ic_key_codes.h"  

    "include/ic/util/profiler.h"

    "include/ic/platform/glfw_input.h" 
    "include/ic/platform/glfw_context.h" 

    "include/ic/interface/layer.h" 
    "include/ic/core/events.h"
    "include/ic/interface/window.h" 
    "include/ic/interface/input.h" 

    "include/ic/renderer/renderer_specification.h" 
    "include/ic/renderer/graphics_api.h" 
    "include/ic/renderer/render_types.h"
    "include/ic/renderer/render_world.h"
    "include/ic/renderer/frame_graph/frame_graph.h" 
    "include/ic/renderer/vulkan_backend/vulkan_backend.h" 
    "include/ic/renderer/frame_graph/frame_graph_node.h"
    "include/ic/renderer/renderer_backend.h"
    "include/ic/renderer/render_handles.h" 
    "include/ic/renderer/renderer_path/renderer_path.h" 
    "include/ic/renderer/renderer_path/deferred_path.h"
    "include/ic/core/job_system.h" 

    "src/ic/core/app_base.cpp" 
    "src/ic/core/layer_stack.cpp" 
    "src/ic/core/event_queue.cpp"
    "src/ic/core/event_frame_buffer.cpp" 
    "src/ic/core/frame_executor.cpp"

    "src/ic/renderer/graphics_api.cpp"
    "src/ic/platform/glfw_input.cpp" 
    "src/ic/platform/glfw_window.cpp" 
    "src/ic/renderer/vulkan_backend/vulkan_backend.cpp" 
    "src/ic/renderer/frame_graph/frame_graph.cpp"
    "src/ic/renderer/renderer_backend.cpp"
    
    "src/ic/core/job_system.cpp" "include/ic/core/clock.h" "include/ic/common/ic_fwd.h" "include/ic/renderer/frame_graph/frame_graph_compiler.h" "src/ic/renderer/frame_graph/frame_graph_compiler.cpp" "include/ic/core/frame_arena.h" "include/ic/core/thread_arena.h")

# Define Visibility and Include Paths
target_include_directories(ImageContinuumEngine
    PUBLIC 
        # Anything linking to the engine (like the demo) can include headers from here.
        # $<BUILD_INTERFACE:...> 
        # prevents the path from bleeding into the install step later.
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Setup PCH
target_precompile_headers(ImageContinuumEngine
    PRIVATE 
        "include/ic/common/ic_pch.h"
)

target_compile_options(ImageContinuumEngine PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/WX>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Werror>
)

# Link against things
target_link_libraries(ImageContinuumEngine
    PUBLIC
        ic_thirdparty

    PRIVATE
        ic_compiler_warnings
)