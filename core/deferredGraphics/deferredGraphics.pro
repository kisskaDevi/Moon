TEMPLATE = lib
CONFIG += static
CONFIG += c++17 console
WARNINGS += -Wall

INCLUDEPATH += \
    $$PWD/../../dependences/libs/vulkan/include/vulkan \
    $$PWD/workflows \
    $$PWD/renderStages \
    $$PWD/../utils \
    $$PWD/../math \
    $$PWD/../interfaces \
    $$PWD/../graphicsManager

SOURCES += \
    deferredGraphics.cpp \
    workflows/blur.cpp \
    workflows/layersCombiner.cpp \
    workflows/postProcessing.cpp \
    workflows/shadow.cpp \
    workflows/skybox.cpp \
    workflows/customFilter.cpp \
    workflows/ssao.cpp \
    workflows/sslr.cpp \
    workflows/workflow.cpp \
    renderStages/source/ambientLighting.cpp \
    renderStages/source/lighting.cpp \
    renderStages/source/lightingPipelines.cpp \
    renderStages/source/base.cpp \
    renderStages/source/extension.cpp \
    renderStages/graphics.cpp \

HEADERS += \
    deferredGraphics.h \
    workflows/blur.h \
    workflows/layersCombiner.h \
    workflows/postProcessing.h \
    workflows/shadow.h \
    workflows/skybox.h \
    workflows/customFilter.h \
    workflows/ssao.h \
    workflows/sslr.h \
    workflows/workflow.h \
    renderStages/graphics.h \

DISTFILES += \
    $$PWD/CMakelists.txt
