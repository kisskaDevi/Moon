cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

set(INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GLTF_DIR}
    ${PLY_DIR}/source
)

include_directories(${INCLUDE_DIRS})

set(SOURCES
    gltfmodel/nodes.cpp
    gltfmodel/gltfmodel.cpp
    gltfmodel/animation.cpp
    plymodel/plymodel.cpp
)

set(HEADERS
    gltfmodel.h
    plymodel.h
)

add_library(models STATIC ${SOURCES} ${HEADERS})
target_include_directories(models PUBLIC ${INCLUDE_DIRS})
target_link_libraries(models PUBLIC interfaces utils)
