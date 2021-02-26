cmake_minimum_required(VERSION 2.8.11)
project(brainfuckTile)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "CMAKE_BUILD_TYPE not set; setting to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()
message(STATUS "Build type is: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS "-march=native -Wall -Wextra -g -funroll-loops")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-Og")

# intgemm:
add_subdirectory(3rd_party/intgemm/)
include_directories(${CMAKE_BINARY_DIR}/3rd_party/intgemm/)
include_directories(3rd_party/intgemm/intgemm)

add_executable(demo.out src/demo.cpp)