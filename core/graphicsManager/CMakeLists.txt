cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall")

include_directories(${VULKAN_INCLUDE_DIR}
${GLFW_DIR}/include/GLFW
../utils
)
                    
set(SOURCES graphicsManager.cpp)

set(HEADERS graphicsInterface.h
graphicsManager.h)

add_library(graphicsManager STATIC ${SOURCES} ${HEADERS})