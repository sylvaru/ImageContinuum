# demo/CMakeLists.txt

add_executable(ImageContinuumDemo 
    "src/main.cpp"
    "include/game_layer.h"
    "include/demo_app.h" 
    "src/demo_app.cpp" 
    "src/game_layer.cpp")

# Set C++ standard to C++20 required
set_property(TARGET ImageContinuumDemo PROPERTY CXX_STANDARD 20)
set_property(TARGET ImageContinuumDemo PROPERTY CXX_STANDARD_REQUIRED ON)

# Tell the demo executable where to find its own local include folder
target_include_directories(ImageContinuumDemo 
    PRIVATE 
        include
)

# Setup PCH
target_precompile_headers(ImageContinuumDemo
    PRIVATE 
        "include/common/demo_pch.h"
)

# Link the demo against the Engine target
target_link_libraries(ImageContinuumDemo 
    PRIVATE 
        ImageContinuumEngine
        ic_compiler_warnings
        spdlog
)