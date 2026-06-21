# cmake/CompilerWarnings.cmake

add_library(ic_compiler_warnings INTERFACE)

if(MSVC)
    target_compile_options(ic_compiler_warnings INTERFACE
        /W4
        /w14242
        /w14254
        /w14263
        /w14265
        /w14287
        /permissive-

        # External headers
        /external:anglebrackets
        /external:W0
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang|GNU")
    target_compile_options(ic_compiler_warnings INTERFACE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
        -Wcast-align
        -Wformat=2
    )
endif()