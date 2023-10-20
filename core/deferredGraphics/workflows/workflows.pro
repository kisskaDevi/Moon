TEMPLATE = lib
CONFIG += static
CONFIG += c++17 console
WARNINGS += -Wall

INCLUDEPATH += \
    $$PWD/../../../dependences/libs/vulkan/include/vulkan \
    $$PWD/../../utils \
    $$PWD/../../math \
    $$PWD/../../interfaces

SOURCES += \
    blur.cpp \
    boundingBox.cpp \
    postProcessing.cpp \
    scattering.cpp \
    shadow.cpp \
    skybox.cpp \
    customFilter.cpp \
    ssao.cpp \
    sslr.cpp \
    workflow.cpp

HEADERS += \
    blur.h \
    boundingBox.h \
    postProcessing.h \
    scattering.h \
    shadow.h \
    skybox.h \
    customFilter.h \
    ssao.h \
    sslr.h \
    workflow.h

DISTFILES += \
    $$PWD/CMakelists.txt
