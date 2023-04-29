cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall")

include_directories(${VULKAN_INCLUDE_DIR}
${GLFW_DIR}/include/GLFW
${STB_DIR}
${GLTF_DIR})
                    
set(SOURCES attachments.cpp
node.cpp
operations.cpp
texture.cpp
device.cpp
vkdefault.cpp
buffer.cpp)

set(HEADERS attachments.h
node.h
operations.h
texture.h
device.h
vkdefault.h
buffer.h)

add_library(utils STATIC ${SOURCES} ${HEADERS})
