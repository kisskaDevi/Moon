cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

include_directories(filters renderStages)

set(SOURCES deferredGraphics.cpp
filters/blur.cpp
filters/filtergraphics.cpp
filters/layersCombiner.cpp
filters/postProcessing.cpp
filters/shadow.cpp
filters/skybox.cpp
filters/customfilter.cpp
filters/ssao.cpp
filters/sslr.cpp
renderStages/source/ambientLighting.cpp
renderStages/source/lighting.cpp
renderStages/source/lightingPipelines.cpp
renderStages/source/base.cpp
renderStages/source/extension.cpp
renderStages/graphics.cpp)

set(HEADERS deferredGraphics.h
filters/blur.h
filters/filtergraphics.h
filters/layersCombiner.h
filters/postProcessing.h
filters/shadow.h
filters/skybox.h
filters/customfilter.h
filters/ssao.h
filters/sslr.h
renderStages/graphics.h)

add_library(deferredGraphics STATIC ${SOURCES} ${HEADERS})
target_include_directories(deferredGraphics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
${CMAKE_CURRENT_SOURCE_DIR}/renderStages
${CMAKE_CURRENT_SOURCE_DIR}/filters)
target_link_libraries(deferredGraphics PUBLIC graphicsManager interfaces utils)
