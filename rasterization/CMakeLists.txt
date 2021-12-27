cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

project(rasterizer VERSION 0.1 LANGUAGES CXX)

if(SKBUILD)
    # Building inside scikit-build system, use python-provided pybind11

    # Scikit-Build does not add your site-packages to the search path
    # automatically, so we need to add
    execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" -c
                "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _tmp_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
    list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
    find_package(pybind11 CONFIG REQUIRED)
else()
    # Use vendored pybind11 when building outside of setup.py
    add_subdirectory(../third_party/pybind11 ${CMAKE_CURRENT_BINARY_DIR}/third_party/pybind11)
endif()

find_package(Vulkan REQUIRED)
find_program(glslc_executable NAMES glslc HINTS Vulkan::glslc)
find_program(xxd_executable NAMES xxd)

find_package(Threads REQUIRED)

if (xxd_executable STREQUAL "xxd_executable-NOTFOUND" OR glslc_executable STREQUAL "glslc_executable-NOTFOUND")
    message(WARNING "Could not find glslc or xxd, disabling shader compilation.")
    set(ENABLE_SHADER_COMPILATION 0)
else()
    set(ENABLE_SHADER_COMPILATION 1)
endif()

function(add_spirv_shader INPUT_FILE)
    if (NOT ENABLE_SHADER_COMPILATION)
        return()
    endif()

    set(COMPILED_SHADER ${CMAKE_CURRENT_SOURCE_DIR}/${INPUT_FILE}.spv)
    set(COMPILED_SHADER_HEADER ${COMPILED_SHADER}.h)

    get_filename_component(SHADER_DIRECTORY ${COMPILED_SHADER} DIRECTORY)
    get_filename_component(SHADER_BASENAME ${COMPILED_SHADER} NAME)

    add_custom_command(
        OUTPUT ${COMPILED_SHADER}
        COMMAND ${glslc_executable} -o ${COMPILED_SHADER} ${INPUT_FILE}
        DEPENDS ${INPUT_FILE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Compiling ${INPUT_FILE} to SPIR-V")

    add_custom_command(
        OUTPUT ${COMPILED_SHADER_HEADER}
        COMMAND ${xxd_executable} -i ${SHADER_BASENAME} ${COMPILED_SHADER_HEADER}
        DEPENDS ${COMPILED_SHADER}
        COMMENT "Generating ${INPUT_FILE}.spv.h"
        WORKING_DIRECTORY ${SHADER_DIRECTORY}
    )

    list(APPEND SPV_SHADERS ${COMPILED_SHADER})
    set(SPV_SHADERS ${SPV_SHADERS} PARENT_SCOPE)

    list(APPEND SPV_SHADERS_HEADERS ${COMPILED_SHADER_HEADER})
    set(SPV_SHADERS_HEADERS ${SPV_SHADERS_HEADERS} PARENT_SCOPE)
endfunction()

add_spirv_shader(shaders/triangle.vert)
add_spirv_shader(shaders/triangle.frag)
add_custom_target(shaders DEPENDS ${SPV_SHADERS} ${SPV_SHADERS_HEADERS})

add_executable(rasterizer src/cpp/main.cpp src/cpp/vulkan_support.cpp src/cpp/point_renderer.cpp src/cpp/lodepng.cpp)
target_link_libraries(rasterizer Vulkan::Vulkan ${CMAKE_DL_LIBS} Threads::Threads)
target_include_directories(rasterizer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

pybind11_add_module(_impl src/cpp/vulkan_support.cpp src/cpp/point_renderer.cpp src/cpp/pybind.cpp)
target_link_libraries(_impl PRIVATE Vulkan::Vulkan ${CMAKE_DL_LIBS} Threads::Threads)
target_include_directories(_impl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS _impl DESTINATION .)
