TEMPLATE = app
CONFIG += c++17 console
WARNINGS += -Wall

Release:DESTDIR = release
Debug:DESTDIR = debug

win32: LIBS += \
    -L$$OUT_PWD/../core/graphicsManager/$$DESTDIR \
    -L$$OUT_PWD/../core/deferredGraphics/$$DESTDIR \
    -L$$OUT_PWD/../core/models/$$DESTDIR \
    -L$$OUT_PWD/../core/transformational/$$DESTDIR \
    -L$$OUT_PWD/../core/interfaces/$$DESTDIR \
    -L$$OUT_PWD/../core/utils/$$DESTDIR \
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
