cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall")

include_directories(${VULKAN_INCLUDE_DIR}
${GLM_DIR}
filters
renderStages
../utils
../interfaces
../graphicsManager
)
                    
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