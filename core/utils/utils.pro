TEMPLATE = lib
CONFIG += static
CONFIG += c++17 console
WARNINGS += -Wall

INCLUDEPATH += \
    $$PWD/../../dependences/libs/vulkan \
    $$PWD/../../dependences/libs/glfw-3.3.4.bin.WIN64/include/GLFW \
    $$PWD/../../dependences/libs/stb \
    $$PWD/../../dependences/libs/tinygltf

SOURCES += \
    attachments.cpp \
    buffer.cpp \
    node.cpp \
    operations.cpp \
    texture.cpp \
    device.cpp \
    vkdefault.cpp

HEADERS += \
    attachments.h \
    buffer.h \
    node.h \
    operations.h \
    texture.h \
    device.h \
    vkdefault.h


DISTFILES += \
    $$PWD/CMakelists.txt
