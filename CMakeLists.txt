cmake_minimum_required(VERSION 3.24)
project(ImageContinuum LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options

option(IC_BUILD_TESTS "Build tests" ON)
option(IC_BUILD_EDITOR "Build editor" OFF)
option(IC_BUILD_SHARED "Build libraries as shared" OFF)

# Setup the scalable compiler warnings module
include(cmake/CompilerWarnings.cmake)


# Add external dependencies first
add_subdirectory(vendor)

add_subdirectory(engine)
add_subdirectory(demo)

if(IC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()






