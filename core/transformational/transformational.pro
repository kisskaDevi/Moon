TEMPLATE = lib
CONFIG += staticlib
CONFIG += c++17 console
WARNINGS += -Wall

INCLUDEPATH += \
    $$PWD/../../dependences/libs/vulkan/include/vulkan \
    $$PWD/../../dependences/libs/glm/glm \
    $$PWD/../utils \
    $$PWD/../interfaces \
    $$PWD/../math

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

DISTFILES += \
    $$PWD/CMakelists.txt
