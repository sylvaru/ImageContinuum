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
    "include/ic/core/frame_context.h"
    "include/ic/core/frame_pipeline.h"

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
     
    "include/ic/renderer/render_types.h"
    "include/ic/renderer/render_world.h"
     
    "include/ic/renderer/vulkan_backend/vulkan_backend.h" 
    
    "include/ic/renderer/renderer_backend.h"
    "include/ic/renderer/render_handles.h" 
    "include/ic/renderer/renderer_path/renderer_path.h" 
    "include/ic/renderer/renderer_path/forward_path.h"
    "include/ic/core/job_system.h" 
    "include/ic/util/util.h"
    "include/ic/core/memory/frame_arena_region.h"
    "include/ic/core/memory/frame_memory_types.h"
    
    "include/ic/renderer/frame_graph/frame_graph_types.h" 
    "include/ic/renderer/frame_graph/frame_graph_pass.h"
    "include/ic/core/clock.h" 
    "include/ic/common/ic_fwd.h" 
    "include/ic/core/memory/frame_arena.h" 
    "include/ic/core/memory/thread_arena.h" 
    "include/ic/core/memory/frame_arena_manager.h"
    "include/ic/core/memory/frame_arena_storage.h"
    "include/ic/renderer/frame_graph/frame_graph_compiler.h"

    "src/ic/core/app_base.cpp" 
    "src/ic/core/layer_stack.cpp" 
    "src/ic/core/event_queue.cpp"
     
    "src/ic/core/frame_pipeline.cpp"

    
    "src/ic/platform/glfw_input.cpp" 
    "src/ic/platform/glfw_window.cpp" 
    "src/ic/renderer/vulkan_backend/vulkan_backend.cpp" 
    
    
    
    "src/ic/core/job_system.cpp"
    "src/ic/renderer/frame_graph/frame_graph_compiler.cpp"
    "src/ic/core/memory/frame_arena.cpp" 

    "include/ic/renderer/frame_graph/frame_graph_builder.h" 
    "src/ic/renderer/frame_graph/frame_graph_builder.cpp"  
    "include/ic/renderer/frame_graph/compiled_graph_plan.h" 
    "include/ic/renderer/frame_graph/frame_graph_cache.h"
    "include/ic/renderer/renderer.h"
    "src/ic/renderer/renderer.cpp" 
    "include/ic/renderer/frame_graph/frame_graph_arena.h"
    "include/ic/renderer/renderer_types.h"  
    "include/ic/renderer/vulkan_backend/vulkan_platform.h" 
    "include/ic/renderer/vulkan_backend/vulkan_adapter.h"
    "include/ic/renderer/vulkan_backend/vulkan_device.h"
    "include/ic/renderer/vulkan_backend/vulkan_allocator.h"
    "include/ic/renderer/vulkan_backend/vulkan_swapchain.h"
    "include/ic/renderer/vulkan_backend/vulkan_command_system.h"

    "src/ic/renderer/vulkan_backend/vulkan_platform.cpp" 
    "src/ic/renderer/vulkan_backend/vulkan_adapter.cpp"
    "src/ic/renderer/vulkan_backend/vulkan_device.cpp"
    "src/ic/renderer/vulkan_backend/vulkan_allocator.cpp"
    "src/ic/renderer/vulkan_backend/vulkan_swapchain.cpp"
    "src/ic/renderer/vulkan_backend/vulkan_command_system.cpp"
    "src/ic/renderer/vulkan_backend/vulkan_instance.cpp"
    "include/ic/renderer/vulkan_backend/vulkan_instance.h"

)

    if (WIN32)
    target_sources(ImageContinuumEngine PRIVATE
    "include/ic/renderer/dx12_backend/dx12_backend.h"
    "include/ic/renderer/dx12_backend/dx12_factory.h"
    "include/ic/renderer/dx12_backend/dx12_adapter.h"
    "include/ic/renderer/dx12_backend/dx12_swapchain.h"
    "include/ic/renderer/dx12_backend/dx12_command_system.h"
    "include/ic/renderer/dx12_backend/dx12_allocator.h"
    )
    endif()

# Defines
target_compile_definitions(ImageContinuumEngine
    PRIVATE
        $<$<CONFIG:Debug>:IC_DEBUG>
)

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