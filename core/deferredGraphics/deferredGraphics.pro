TEMPLATE = lib
CONFIG += static
CONFIG += c++17 console
WARNINGS += -Wall

INCLUDEPATH += \
    $$PWD/../../dependences/libs/vulkan/include/vulkan \
    $$PWD/renderStages \
    $$PWD/workflows \
    $$PWD/../utils \
    $$PWD/../math \
    $$PWD/../interfaces \
    $$PWD/../graphicsManager

SOURCES += \
    deferredGraphics.cpp \
    renderStages/deferredAttachments.cpp \
    renderStages/link.cpp \
    renderStages/source/ambientLighting.cpp \
    renderStages/source/lighting.cpp \
    renderStages/source/lightingPipelines.cpp \
    renderStages/source/base.cpp \
    renderStages/source/extension.cpp \
    renderStages/graphics.cpp \
    renderStages/layersCombiner.cpp

HEADERS += \
    deferredGraphics.h \
    renderStages/deferredAttachments.h \
    renderStages/link.h \
    renderStages/graphics.h \
    renderStages/layersCombiner.h

DISTFILES += \
    $$PWD/CMakelists.txt
