cmake_minimum_required(VERSION 3.5)

project(kisskaVulkan LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall")

add_subdirectory(core/utils)
add_subdirectory(core/graphicsManager)
add_subdirectory(core/interfaces)
add_subdirectory(core/deferredGraphics)
add_subdirectory(core/models)
add_subdirectory(core/transformational)
add_subdirectory(test)