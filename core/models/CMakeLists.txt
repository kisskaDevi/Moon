cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(SOURCES gltfmodel/nodes.cpp
gltfmodel/gltfmodel.cpp
gltfmodel/animation.cpp
plymodel/plymodel.cpp)

set(HEADERS gltfmodel.h plymodel.h)

add_library(models STATIC ${SOURCES} ${HEADERS})
target_include_directories(models PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(models PUBLIC interfaces utils)
