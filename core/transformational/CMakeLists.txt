cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

set(SOURCES
    baseCamera.cpp
    baseObject.cpp
    group.cpp
    spotLight.cpp
)

set(HEADERS
    baseCamera.h
    baseObject.h
    spotLight.h
    transformational.h
    group.h
)

add_library(transformational STATIC ${SOURCES} ${HEADERS})
target_include_directories(transformational PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(transformational PUBLIC math interfaces utils)
