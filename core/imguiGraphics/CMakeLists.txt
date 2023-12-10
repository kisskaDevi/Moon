cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

include_directories(${IMGUI_DIR}
${IMGUI_DIR}/backends
${Vulkan_INCLUDE_DIRS}
${GLFW_DIR}/include)

set(SOURCES imguiGraphics.cpp
imguiLink.cpp
${IMGUI_DIR}/imgui.cpp
${IMGUI_DIR}/imgui_draw.cpp
${IMGUI_DIR}/imgui_tables.cpp
${IMGUI_DIR}/imgui_widgets.cpp
${IMGUI_DIR}/imgui_demo.cpp
${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
${IMGUI_DIR}/backends/imgui_impl_glfw.cpp)

set(HEADERS imguiGraphics.h
imguiLink.h
${IMGUI_DIR}/imgui.h
${IMGUI_DIR}/backends/imgui_impl_vulkan.h
${IMGUI_DIR}/backends/imgui_impl_glfw.h)

add_library(imguiGraphics STATIC ${SOURCES} ${HEADERS})
target_include_directories(imguiGraphics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
${IMGUI_DIR}
${IMGUI_DIR}/backends
${Vulkan_INCLUDE_DIRS})
target_link_libraries(imguiGraphics PUBLIC graphicsManager utils)
