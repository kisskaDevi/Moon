TEMPLATE = app
CONFIG += c++17 console
WARNINGS += -Wall

Release:DESTDIR = release
Debug:DESTDIR = debug

Release:GLFW_BIN_DIR = Release
Debug:GLFW_BIN_DIR = Debug

win32: LIBS += \
    -L$$OUT_PWD/../core/graphicsManager/$$DESTDIR \
    -L$$OUT_PWD/../core/deferredGraphics/$$DESTDIR \
    -L$$OUT_PWD/../core/models/$$DESTDIR \
    -L$$OUT_PWD/../core/transformational/$$DESTDIR \
    -L$$OUT_PWD/../core/interfaces/$$DESTDIR \
    -L$$OUT_PWD/../core/utils/$$DESTDIR \
    -L$$PWD/../dependences/libs/vulkan_runtime/x64 \
    -L$$PWD/../dependences/libs/glfw/build/src/$$GLFW_BIN_DIR \
    -lgraphicsManager \
    -ldeferredGraphics \
    -lmodels \
    -ltransformational \
    -linterfaces \
    -lutils \
    -lvulkan-1 \
    -lglfw3

INCLUDEPATH += \
    $$PWD/../dependences\libs \
    $$PWD/../dependences/libs/vulkan/include/vulkan \
    $$PWD/../dependences/libs/glfw/include/GLFW \
    $$PWD/../dependences/libs/glm/glm \
    $$PWD/../dependences/libs/stb \
    $$PWD/../dependences/libs/tinygltf \
    $$PWD/../core/graphicsManager \
    $$PWD/../core/deferredGraphics \
    $$PWD/../core/deferredGraphics/renderStages \
    $$PWD/../core/deferredGraphics/filters \
    $$PWD/../core/utils \
    $$PWD/../core/transformational \
    $$PWD/../core/interfaces \
    $$PWD/../core/models \
    $$PWD/../core/math

SOURCES += \
    physicalobject.cpp \
    scene.cpp \
    main.cpp

HEADERS += \
    physicalobject.h \
    scene.h

DISTFILES += \
    $$PWD/CMakelists.txt
