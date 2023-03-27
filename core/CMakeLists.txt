cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall")

include_directories(../core
../dependences/libs
${VULKAN_INCLUDE_DIR}
${GLFW_DIR}/include/GLFW
${GLM_DIR}
${STB_DIR}
${GLTF_DIR})
                    
set(SOURCES deferredGraphics/deferredGraphics.cpp
deferredGraphics/filters/blur.cpp
deferredGraphics/filters/filtergraphics.cpp
deferredGraphics/filters/layersCombiner.cpp
deferredGraphics/filters/postProcessing.cpp
deferredGraphics/filters/shadow.cpp
deferredGraphics/filters/skybox.cpp
deferredGraphics/filters/customfilter.cpp
deferredGraphics/filters/ssao.cpp
deferredGraphics/filters/sslr.cpp
deferredGraphics/renderStages/source/ambientLighting.cpp
deferredGraphics/renderStages/source/lighting.cpp
deferredGraphics/renderStages/source/lightingPipelines.cpp
deferredGraphics/renderStages/source/base.cpp
deferredGraphics/renderStages/source/extension.cpp
deferredGraphics/renderStages/graphics.cpp
interfaces/model.cpp
interfaces/light.cpp
transformational/camera.cpp
transformational/group.cpp
transformational/spotLight.cpp
transformational/object.cpp
models/gltfmodel.cpp
utils/attachments.cpp
utils/node.cpp
utils/operations.cpp
utils/texture.cpp
utils/device.cpp
utils/vkdefault.cpp
graphicsManager.cpp)

add_library(core STATIC ${SOURCES})