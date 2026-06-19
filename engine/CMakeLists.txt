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
    "src/core/app_base.cpp" 
     
    "include/image_continuum/core/event_queue.h" 
    "include/image_continuum/interface/layer.h" 
    "include/image_continuum/interface/events.h"
    
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
    

# Link the strict compiler warnings interface
target_link_libraries(ImageContinuumEngine PRIVATE ic_compiler_warnings)