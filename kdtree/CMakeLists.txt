cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

project(kdtree VERSION 0.1 LANGUAGES CXX)

include(CTest)

if(BUILD_TESTING)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/6b74da4757a549563d7c37c8fae3e704662a043b.zip
    )

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()


find_package(Threads REQUIRED)

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


add_library(kdtree STATIC src/cpp/kdtree.cpp)
target_link_libraries(kdtree PRIVATE Threads::Threads)
target_compile_options(kdtree PUBLIC $<$<CONFIG:DEBUG>:-fsanitize=address>)
target_link_options(kdtree PUBLIC $<$<CONFIG:DEBUG>:-fsanitize=address>)
target_include_directories(kdtree PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/misc)
set_property(TARGET kdtree PROPERTY POSITION_INDEPENDENT_CODE ON)

add_executable(kdtree_main src/cpp/main.cpp)
target_link_libraries(kdtree_main PRIVATE kdtree)
target_include_directories(kdtree_main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/random123/include)
target_include_directories(kdtree_main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/misc)


pybind11_add_module(_impl src/cpp/pybind.cpp)
target_link_libraries(_impl PRIVATE kdtree)

if(BUILD_TESTING)
    add_executable(kdtree_test src/cpp/test.cpp)
    target_link_libraries(kdtree_test PRIVATE kdtree gtest_main)

    include(GoogleTest)
    gtest_discover_tests(kdtree_test)
endif()

