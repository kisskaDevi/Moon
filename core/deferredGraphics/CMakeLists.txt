cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

include_directories(workflows renderStages)

set(SOURCES deferredGraphics.cpp
workflows/blur.cpp
workflows/workflow.cpp
workflows/layersCombiner.cpp
workflows/postProcessing.cpp
workflows/shadow.cpp
workflows/skybox.cpp
workflows/customFilter.cpp
workflows/ssao.cpp
workflows/sslr.cpp
workflows/scattering.cpp
renderStages/source/ambientLighting.cpp
renderStages/source/lighting.cpp
renderStages/source/lightingPipelines.cpp
renderStages/source/base.cpp
renderStages/source/extension.cpp
renderStages/graphics.cpp
renderStages/link.cpp)

set(HEADERS deferredGraphics.h
workflows/blur.h
workflows/workflow.h
workflows/layersCombiner.h
workflows/postProcessing.h
workflows/shadow.h
workflows/skybox.h
workflows/customFilter.h
workflows/ssao.h
workflows/sslr.h
workflows/scattering.h
renderStages/graphics.h
renderStages/link.h)

add_library(deferredGraphics STATIC ${SOURCES} ${HEADERS})
target_include_directories(deferredGraphics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
${CMAKE_CURRENT_SOURCE_DIR}/renderStages
${CMAKE_CURRENT_SOURCE_DIR}/workflows)
target_link_libraries(deferredGraphics PUBLIC graphicsManager interfaces utils)
