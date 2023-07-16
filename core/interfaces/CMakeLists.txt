cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

set(SOURCES light.cpp
model.cpp
object.cpp)

set(HEADERS camera.h
light.h
model.h
object.h)

add_library(interfaces STATIC ${SOURCES} ${HEADERS})
target_include_directories(interfaces PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(interfaces PUBLIC utils)
