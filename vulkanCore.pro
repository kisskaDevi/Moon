CONFIG += c++17 console

win32: LIBS += \
    -L$$PWD/dependences/libs/vulkan/x64 \
    -L$$PWD/dependences/libs/glfw-3.3.4.bin.WIN64/lib-static-ucrt \
    -lvulkan-1 \
    -lglfw3dll

INCLUDEPATH += \
    $$PWD/dependences/libs \
    $$PWD/dependences/libs/vulkan \
    $$PWD/dependences/libs/glfw-3.3.4.bin.WIN64/include/GLFW \
    $$PWD/dependences/libs/glm \
    $$PWD/dependences/libs/stb \
    $$PWD/dependences/libs/tinygltf

DEPENDPATH += \
    $$PWD/dependences

DISTFILES += \
    core/deferredGraphics/shaders/spotLightingPass/metods/lightDrop.frag \
    core/deferredGraphics/shaders/spotLightingPass/metods/outsideSpotCondition.frag \
    core/deferredGraphics/shaders/spotLightingPass/metods/pbr.frag \
    core/deferredGraphics/shaders/spotLightingPass/metods/scattering.frag \
    core/deferredGraphics/shaders/spotLightingPass/metods/shadow.frag \
    core/deferredGraphics/shaders/spotLightingPass/scatteringShadowSpotLighting.frag \
    core/deferredGraphics/shaders/spotLightingPass/scatteringSpotLighting.frag \
    core/deferredGraphics/shaders/spotLightingPass/shadowSpotLighting.frag \
    core/deferredGraphics/shaders/spotLightingPass/spotLighting.frag \
    core/deferredGraphics/shaders/spotLightingPass/spotLighting.vert \
    core/deferredGraphics/shaders/ambientLightingPass/ambientLighting.frag \
    core/deferredGraphics/shaders/ambientLightingPass/ambientLighting.vert \
    core/deferredGraphics/shaders/compile.bat \
    core/deferredGraphics/shaders/base/base.frag \
    core/deferredGraphics/shaders/base/base.vert \
    core/deferredGraphics/shaders/customFilter/customFilter.frag \
    core/deferredGraphics/shaders/customFilter/customFilter.vert \
    core/deferredGraphics/shaders/gaussianBlur/xBlur.frag \
    core/deferredGraphics/shaders/gaussianBlur/xBlur.vert \
    core/deferredGraphics/shaders/gaussianBlur/yBlur.frag \
    core/deferredGraphics/shaders/gaussianBlur/yBlur.vert \
    core/deferredGraphics/shaders/layersCombiner/layersCombiner.frag \
    core/deferredGraphics/shaders/layersCombiner/layersCombiner.vert \
    core/deferredGraphics/shaders/shadow/shadowMapShader.vert \
    core/deferredGraphics/shaders/postProcessing/postProcessingShader.frag \
    core/deferredGraphics/shaders/postProcessing/postProcessingShader.vert \
    core/deferredGraphics/shaders/skybox/skybox.frag \
    core/deferredGraphics/shaders/skybox/skybox.vert \
    core/deferredGraphics/shaders/ssao/SSAO.frag \
    core/deferredGraphics/shaders/ssao/SSAO.vert \
    core/deferredGraphics/shaders/sslr/SSLR.frag \
    core/deferredGraphics/shaders/sslr/SSLR.vert \
    core/deferredGraphics/shaders/outlining/outlining.frag \
    core/deferredGraphics/shaders/outlining/outlining.vert

SOURCES += \
    core/deferredGraphics/deferredGraphics.cpp \
    core/deferredGraphics/filters/blur.cpp \
    core/deferredGraphics/filters/filtergraphics.cpp \
    core/deferredGraphics/filters/layersCombiner.cpp \
    core/deferredGraphics/filters/postProcessing.cpp \
    core/deferredGraphics/filters/shadow.cpp \
    core/deferredGraphics/filters/skybox.cpp \
    core/deferredGraphics/filters/customfilter.cpp \
    core/deferredGraphics/filters/ssao.cpp \
    core/deferredGraphics/filters/sslr.cpp \
    core/deferredGraphics/renderStages/source/ambientLighting.cpp \
    core/deferredGraphics/renderStages/source/lighting.cpp \
    core/deferredGraphics/renderStages/source/lightingPipelines.cpp \
    core/deferredGraphics/renderStages/source/base.cpp \
    core/deferredGraphics/renderStages/source/extension.cpp \
    core/deferredGraphics/renderStages/graphics.cpp \
    core/interfaces/light.cpp \
    core/interfaces/model.cpp \
    core/transformational/camera.cpp \
    core/transformational/group.cpp \
    core/transformational/object.cpp \
    core/transformational/spotLight.cpp \
    core/models/gltfmodel.cpp \
    core/utils/attachments.cpp \
    core/utils/node.cpp \
    core/utils/operations.cpp \
    core/utils/texture.cpp \
    core/utils/device.cpp \
    core/utils/vkdefault.cpp \
    core/graphicsManager.cpp\
    test/physicalobject.cpp \
    test/scene.cpp \
    test/main.cpp

HEADERS += \
    core/deferredGraphics/deferredGraphics.h \
    core/deferredGraphics/filters/blur.h \
    core/deferredGraphics/filters/filtergraphics.h \
    core/deferredGraphics/filters/layersCombiner.h \
    core/deferredGraphics/filters/postProcessing.h \
    core/deferredGraphics/filters/shadow.h \
    core/deferredGraphics/filters/skybox.h \
    core/deferredGraphics/filters/customfilter.h \
    core/deferredGraphics/filters/ssao.h \
    core/deferredGraphics/filters/sslr.h \
    core/deferredGraphics/renderStages/graphics.h \
    core/interfaces/light.h \
    core/interfaces/model.h \
    core/transformational/spotLight.h \
    core/transformational/transformational.h \
    core/transformational/camera.h \
    core/transformational/group.h \
    core/transformational/object.h \
    core/models/gltfmodel.h \
    core/utils/attachments.h \
    core/utils/node.h \
    core/utils/operations.h \
    core/utils/texture.h \
    core/utils/device.h \
    core/utils/vkdefault.h \
    core/graphicsInterface.h \
    core/graphicsManager.h \
    test/physicalobject.h \
    test/scene.h
