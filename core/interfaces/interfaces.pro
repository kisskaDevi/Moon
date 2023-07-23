TEMPLATE = lib
CONFIG += staticlib
CONFIG += c++17 console
WARNINGS += -Wall

INCLUDEPATH += \
    $$PWD/../../dependences/libs/vulkan/include/vulkan \
    $$PWD/../utils \
    $$PWD/../math

SOURCES += \
    light.cpp \
    model.cpp \
    object.cpp

HEADERS += \
    camera.h \
    light.h \
    model.h \
    object.h

DISTFILES += \
    $$PWD/CMakelists.txt