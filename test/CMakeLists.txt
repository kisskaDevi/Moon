cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()
set(CMAKE_DEBUG_POSTFIX d)

set(SOURCES main.cpp testScene.cpp controller.cpp)
set(INCLUDES scene.h testScene.h controller.h)

add_executable(test ${INCLUDES} ${SOURCES})

target_link_libraries(test PRIVATE deferredGraphics imguiGraphics models transformational ${LIBS_DIRS})

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    add_custom_command(TARGET test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<$<CONFIG:Debug>:${GLFW_DIR}/build/src/DebugRelease/glfw3.dll>
    $<$<CONFIG:Release>:${GLFW_DIR}/build/src/Release/glfw3.dll>
    $<TARGET_FILE_DIR:test>)
endif()

