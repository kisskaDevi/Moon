TEMPLATE = lib
CONFIG += staticlib
CONFIG += c++17 console
WARNINGS += -Wall

INCLUDEPATH += \
    $$PWD/../../dependences/libs/vulkan \
    $$PWD/../../dependences/libs/glm \
    $$PWD/../../dependences/libs/stb \
    $$PWD/../../dependences/libs/tinygltf \
    $$PWD/../interfaces \
    $$PWD/../utils \
    $$PWD

SOURCES += \
    gltfmodel/nodes.cpp \
    gltfmodel/gltfmodel.cpp \
    gltfmodel/animation.cpp

HEADERS += \
    gltfmodel.h
