cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall")

set(INCLUDE_DIRS ${Vulkan_INCLUDE_DIRS} ${GLFW_DIR}/include/GLFW
${STB_DIR} ${GLTF_DIR} ${GLM_DIR}/glm)

include_directories(${INCLUDE_DIRS})

set(SOURCES attachments.cpp
node.cpp
operations.cpp
texture.cpp
device.cpp
vkdefault.cpp
buffer.cpp
swapChain.cpp)

set(HEADERS attachments.h
node.h
operations.h
texture.h
device.h
vkdefault.h
buffer.h
swapChain.h)

add_library(utils STATIC ${SOURCES} ${HEADERS})
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${INCLUDE_DIRS})
target_link_libraries(utils PUBLIC ${LIBS_DIRS})
