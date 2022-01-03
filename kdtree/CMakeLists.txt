cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/6b74da4757a549563d7c37c8fae3e704662a043b.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

project(kdtree VERSION 0.1 LANGUAGES CXX)

find_package(Threads REQUIRED)


add_library(kdtree STATIC src/cpp/kdtree.cpp)
target_link_libraries(kdtree PRIVATE Threads::Threads)
target_compile_options(kdtree PUBLIC $<$<CONFIG:DEBUG>:-fsanitize=address>)
target_link_options(kdtree PUBLIC $<$<CONFIG:DEBUG>:-fsanitize=address>)
target_include_directories(kdtree PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/misc)

add_executable(kdtree_main src/cpp/main.cpp)
target_link_libraries(kdtree_main PRIVATE kdtree)

add_executable(kdtree_test src/cpp/test.cpp)
target_link_libraries(kdtree_test PRIVATE kdtree gtest_main)

include(GoogleTest)
gtest_discover_tests(kdtree_test)
