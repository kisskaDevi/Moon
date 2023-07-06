cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall")

include_directories(../dependences/libs
${VULKAN_INCLUDE_DIR}
${GLFW_DIR}/include/GLFW
${GLM_DIR}
${STB_DIR}
${GLTF_DIR}
../core/graphicsManager
../core/deferredGraphics
../core/deferredGraphics/renderStages
../core/deferredGraphics/filters
../core/utils
../core/transformational
../core/interfaces
../core/models
../core/math)
                    
set(SOURCES main.cpp scene.cpp)
set(INCLUDES scene.h)

add_executable(test ${INCLUDES} ${SOURCES})

SET(LIBS ${VULKAN_LIB_DIR}/vulkan-1.lib
${GLFW_DIR}/lib-static-ucrt/glfw3dll.lib)

target_link_libraries(test graphicsManager deferredGraphics models transformational interfaces utils ${LIBS})

add_custom_command(TARGET test POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_if_different
"${GLFW_DIR}/lib-static-ucrt/glfw3.dll"
$<TARGET_FILE_DIR:test>)
