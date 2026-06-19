# demo/CMakeLists.txt

add_executable(ImageContinuumDemo 
    "src/main.cpp"
    "include/game_layer.h"
)

# Set C++ standard to C++20 required
set_property(TARGET ImageContinuumDemo PROPERTY CXX_STANDARD 20)
set_property(TARGET ImageContinuumDemo PROPERTY CXX_STANDARD_REQUIRED ON)

# Tell the demo executable where to find its own local include folder
target_include_directories(ImageContinuumDemo 
    PRIVATE 
        include
)

# Link the demo against the Engine target
# This automatically pulls in the engine's libraries and configurations
target_link_libraries(ImageContinuumDemo 
    PRIVATE 
        ImageContinuumEngine
        ic_compiler_warnings
)