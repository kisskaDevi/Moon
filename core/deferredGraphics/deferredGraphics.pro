TEMPLATE = lib
CONFIG += static
CONFIG += c++17 console
WARNINGS += -Wall

INCLUDEPATH += \
    $$PWD/../../dependences/libs/vulkan/include/vulkan \
    $$PWD/filters \
    $$PWD/renderStages \
    $$PWD/../utils \
    $$PWD/../math \
    $$PWD/../interfaces \
    $$PWD/../graphicsManager

DISTFILES += \
    shaders/spotLightingPass/metods/lightDrop.frag \
    shaders/spotLightingPass/metods/outsideSpotCondition.frag \
    shaders/spotLightingPass/metods/pbr.frag \
    shaders/spotLightingPass/metods/scattering.frag \
    shaders/spotLightingPass/metods/shadow.frag \
    shaders/spotLightingPass/scatteringShadowSpotLighting.frag \
    shaders/spotLightingPass/scatteringSpotLighting.frag \
    shaders/spotLightingPass/shadowSpotLighting.frag \
    shaders/spotLightingPass/spotLighting.frag \
    shaders/spotLightingPass/spotLighting.vert \
    shaders/ambientLightingPass/ambientLighting.frag \
    shaders/ambientLightingPass/ambientLighting.vert \
    shaders/compile.bat \
    shaders/base/base.frag \
    shaders/base/base.vert \
    shaders/customFilter/customFilter.frag \
    shaders/customFilter/customFilter.vert \
    shaders/gaussianBlur/xBlur.frag \
    shaders/gaussianBlur/xBlur.vert \
    shaders/gaussianBlur/yBlur.frag \
    shaders/gaussianBlur/yBlur.vert \
    shaders/layersCombiner/layersCombiner.frag \
    shaders/layersCombiner/layersCombiner.vert \
    shaders/shadow/shadowMapShader.vert \
    shaders/postProcessing/postProcessingShader.frag \
    shaders/postProcessing/postProcessingShader.vert \
    shaders/skybox/skybox.frag \
    shaders/skybox/skybox.vert \
    shaders/ssao/SSAO.frag \
    shaders/ssao/SSAO.vert \
    shaders/sslr/SSLR.frag \
    shaders/sslr/SSLR.vert \
    shaders/outlining/outlining.frag \
    shaders/outlining/outlining.vert

SOURCES += \
    deferredGraphics.cpp \
    filters/blur.cpp \
    filters/filtergraphics.cpp \
    filters/layersCombiner.cpp \
    filters/postProcessing.cpp \
    filters/shadow.cpp \
    filters/skybox.cpp \
    filters/customfilter.cpp \
    filters/ssao.cpp \
    filters/sslr.cpp \
    renderStages/source/ambientLighting.cpp \
    renderStages/source/lighting.cpp \
    renderStages/source/lightingPipelines.cpp \
    renderStages/source/base.cpp \
    renderStages/source/extension.cpp \
    renderStages/graphics.cpp \

HEADERS += \
    deferredGraphics.h \
    filters/blur.h \
    filters/filtergraphics.h \
    filters/layersCombiner.h \
    filters/postProcessing.h \
    filters/shadow.h \
    filters/skybox.h \
    filters/customfilter.h \
    filters/ssao.h \
    filters/sslr.h \
    renderStages/graphics.h \

DISTFILES += \
    $$PWD/CMakelists.txt
