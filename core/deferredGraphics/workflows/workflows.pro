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
    bloom.cpp \
    blur.cpp \
    boundingBox.cpp \
    postProcessing.cpp \
    scattering.cpp \
    selector.cpp \
    shadow.cpp \
    skybox.cpp \
    ssao.cpp \
    sslr.cpp \
    workflow.cpp

HEADERS += \
    bloom.h \
    blur.h \
    boundingBox.h \
    postProcessing.h \
    scattering.h \
    selector.h \
    shadow.h \
    skybox.h \
    ssao.h \
    sslr.h \
    workflow.h

DISTFILES += \
    $$PWD/CMakelists.txt
