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
    "include/image_continuum/core/app_base.h" 
    "include/image_continuum/core/event_queue.h" 
    "include/image_continuum/interface/layer.h" 
    "include/image_continuum/interface/events.h"
    "include/image_continuum/interface/window.h" 
    "include/image_continuum/interface/renderer_backend.h" 
    "include/image_continuum/interface/input.h" 
    "include/image_continuum/common/ic_common.h" 
    "include/image_continuum/common/ic_key_codes.h"  
    "include/image_continuum/platform/glfw_input.h" 
    "include/image_continuum/platform/glfw_context.h" 
    "include/image_continuum/platform/rendering/graphics_api.h" 
    

    "src/core/app_base.cpp" 
    "src/core/layer_stack.cpp" 
    "src/core/event_queue.cpp"
    "src/platform/rendering/graphics_api.cpp"
    "src/platform/glfw_input.cpp" 
    "src/platform/glfw_window.cpp" 
    "src/platform/glfw_window.cpp" 
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
        "include/image_continuum/common/ic_pch.h"
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