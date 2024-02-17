TEMPLATE = lib
CONFIG += staticlib
CONFIG += c++17 console
WARNINGS += -Wall

INCLUDEPATH += \
    $$PWD/../../dependences/libs/vulkan/include/vulkan \
    $$PWD/../../dependences/libs/glfw/include/GLFW \
    $$PWD/../utils \
    $$PWD/../math

SOURCES += \
    graphicsLinker.cpp \
    graphicsManager.cpp

HEADERS += \
    graphicsInterface.h \
    graphicsLinker.h \
    graphicsManager.h \
    linkable.h

DISTFILES += \
    $$PWD/CMakelists.txt
