TEMPLATE = app
CONFIG += c++17 console
WARNINGS += -Wall

win32: LIBS += \
    -L$$OUT_PWD/../core/graphicsManager/debug \
    -L$$OUT_PWD/../core/deferredGraphics/debug \
    -L$$OUT_PWD/../core/models/debug \
    -L$$OUT_PWD/../core/transformational/debug \
    -L$$OUT_PWD/../core/interfaces/debug \
    -L$$OUT_PWD/../core/utils/debug \
    -L$$PWD/../dependences/libs/vulkan/x64 \
    -L$$PWD/../dependences/libs/glfw-3.3.4.bin.WIN64/lib-static-ucrt \
    -lgraphicsManager \
    -ldeferredGraphics \
    -lmodels \
    -ltransformational \
    -linterfaces \
    -lutils \
    -lvulkan-1 \
    -lglfw3dll


INCLUDEPATH += \
    $$PWD/../dependences\libs \
    $$PWD/../dependences/libs/vulkan \
    $$PWD/../dependences/libs/glfw-3.3.4.bin.WIN64/include/GLFW \
    $$PWD/../dependences/libs/glm \
    $$PWD/../dependences/libs/stb \
    $$PWD/../dependences/libs/tinygltf \
    $$PWD/../core/graphicsManager \
    $$PWD/../core/deferredGraphics \
    $$PWD/../core/deferredGraphics/renderStages \
    $$PWD/../core/deferredGraphics/filters \
    $$PWD/../core/utils \
    $$PWD/../core/transformational \
    $$PWD/../core/interfaces \
    $$PWD/../core/models

SOURCES += \
    physicalobject.cpp \
    scene.cpp \
    main.cpp

HEADERS += \
    physicalobject.h \
    scene.h
