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
