cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall")
set(CMAKE_DEBUG_POSTFIX d)

set(SOURCES main.cpp scene.cpp)
set(INCLUDES scene.h)

add_executable(test ${INCLUDES} ${SOURCES})

target_link_libraries(test PRIVATE deferredGraphics models transformational ${LIBS_DIRS})

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    add_custom_command(TARGET test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<$<CONFIG:Debug>:${GLFW_DIR}/build/src/DebugRelease/glfw3.dll>
    $<$<CONFIG:Release>:${GLFW_DIR}/build/src/Release/glfw3.dll>
    $<TARGET_FILE_DIR:test>)
endif()

