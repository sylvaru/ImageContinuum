# cmake/CompilerWarnings.cmake

# Create the phantom interface target
add_library(ic_compiler_warnings INTERFACE)

# Detect the compiler and apply the strict flags
if(MSVC)
    target_compile_options(ic_compiler_warnings INTERFACE
        /W4     # Baseline reasonable strict warnings
        /w14242 # 'identifier': conversion from 'type1' to 'type2', possible loss of data
        /w14254 # 'operator': conversion from 'type1:field_bits' to 'type2:field_bits'
        /w14263 # 'function': member function does not override any base class virtual member function
        /w14265 # 'classname': class has virtual functions, but destructor is not virtual
        /w14287 # 'operator': unsigned/negative constant mismatch
        /permissive- # Enforces strict C++ standard conformance
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang|GNU")
    target_compile_options(ic_compiler_warnings INTERFACE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion        # Warn on type conversions that may lose data
        -Wsign-conversion   # Warn on sign conversions
        -Wcast-align        # Warn on pointer casts that increase alignment requirements
        -Wformat=2          # Warn on security issues around functions that format output
    )
endif()

# Handle Warnings as Errors globally based on CMakePresets.json
if(CMAKE_COMPILE_WARNING_AS_ERROR)
    if(MSVC)
        target_compile_options(ic_compiler_warnings INTERFACE /WX)
    else()
        target_compile_options(ic_compiler_warnings INTERFACE -Werror)
    endif()
endif()