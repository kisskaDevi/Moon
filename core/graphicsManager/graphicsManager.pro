TEMPLATE = lib
CONFIG += staticlib
CONFIG += c++17 console
WARNINGS += -Wall

INCLUDEPATH += \
    $$PWD/../../dependences/libs/vulkan \
    $$PWD/../../dependences/libs/glfw/include/GLFW \
    $$PWD/../utils

SOURCES += \
    graphicsManager.cpp

HEADERS += \
    graphicsInterface.h \
    graphicsManager.h

DISTFILES += \
    $$PWD/CMakelists.txt
