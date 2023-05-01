TEMPLATE = lib
CONFIG += staticlib
CONFIG += c++17 console
WARNINGS += -Wall

INCLUDEPATH += \
    $$PWD/../../dependences/libs/vulkan \
    $$PWD/../../dependences/libs/glfw-3.3.4.bin.WIN64/include/GLFW \
    $$PWD/../utils

SOURCES += \
    graphicsManager.cpp

HEADERS += \
    graphicsInterface.h \
    graphicsManager.h

DISTFILES += \
    $$PWD/CMakelists.txt
