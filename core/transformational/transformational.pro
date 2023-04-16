TEMPLATE = lib
CONFIG += staticlib
CONFIG += c++17 console
WARNINGS += -Wall

INCLUDEPATH += \
    $$PWD/../../dependences/libs \
    $$PWD/../../dependences/libs/vulkan \
    $$PWD/../../dependences/libs/glm \
    $$PWD/../utils \
    $$PWD/../interfaces

SOURCES += \
    baseCamera.cpp \
    baseObject.cpp \
    group.cpp \
    spotLight.cpp

HEADERS += \
    baseCamera.h \
    baseObject.h \
    spotLight.h \
    transformational.h \
    group.h
