TEMPLATE = lib
CONFIG += staticlib
CONFIG += c++17 console
WARNINGS += -Wall

INCLUDEPATH += \
    $$PWD/../../dependences/libs/vulkan/include/vulkan \
    $$PWD/../../dependences/libs/stb \
    $$PWD/../../dependences/libs/tinygltf \
    $$PWD/../../dependences/libs/tinyply/source \
    $$PWD/../interfaces \
    $$PWD/../utils \
    $$PWD/../math \
    $$PWD

SOURCES += \
    gltfmodel/nodes.cpp \
    gltfmodel/gltfmodel.cpp \
    gltfmodel/animation.cpp \
    plymodel/plymodel.cpp

HEADERS += \
    gltfmodel.h \
    plymodel.h

DISTFILES += \
    $$PWD/CMakelists.txt
