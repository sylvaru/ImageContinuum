
add_executable(ImageContinuumTests
    "src/main.cpp"
)

# Set C++ standard to C++20 required
set_property(TARGET ImageContinuumTests PROPERTY CXX_STANDARD 20)
set_property(TARGET ImageContinuumTests PROPERTY CXX_STANDARD_REQUIRED ON)

# Tell the tests executable where to find its local 
# include folder
target_include_directories(ImageContinuumTests
    PRIVATE 
        include
)

# Setup PCH
target_precompile_headers(ImageContinuumTests
    PRIVATE  
         "include/common/tests_pch.h"
)
    
# Link the tests against the Engine target,
# warnings, and any testing frameworks
target_link_libraries(ImageContinuumTests
    PRIVATE 
        ImageContinuumEngine
        ic_compiler_warnings
        spdlog
)