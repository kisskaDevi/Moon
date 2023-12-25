TEMPLATE = lib
CONFIG += static
CONFIG += c++17 console
WARNINGS += -Wall

INCLUDEPATH += \
    $$PWD/../../dependences/libs/vulkan/include/vulkan \
    $$PWD/../../dependences/libs/glfw/include/GLFW \
    $$PWD/../../dependences/libs/stb \
    $$PWD/../../dependences/libs/tinygltf

SOURCES += \
    attachments.cpp \
    buffer.cpp \
    node.cpp \
    operations.cpp \
    swapChain.cpp \
    texture.cpp \
    device.cpp \
    vkdefault.cpp \
    depthMap.cpp

HEADERS += \
    attachments.h \
    buffer.h \
    node.h \
    operations.h \
    swapChain.h \
    texture.h \
    device.h \
    vkdefault.h \
    depthMap.h


DISTFILES += \
    $$PWD/CMakelists.txt
