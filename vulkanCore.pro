CONFIG += c++17 console

win32: LIBS += -L$$PWD/libs/Lib/vulkan/x64/ -lvulkan-1

INCLUDEPATH += $$PWD/libs/Lib/vulkan/x64
DEPENDPATH += $$PWD/libs/Lib/vulkan/x64

win32: LIBS += -L$$PWD/libs/glfw-3.3.4.bin.WIN64/lib-static-ucrt/ -lglfw3dll

INCLUDEPATH += $$PWD/libs/glfw-3.3.4.bin.WIN64/lib-static-ucrt
DEPENDPATH += $$PWD/libs/glfw-3.3.4.bin.WIN64/lib-static-ucrt

DISTFILES += \
    core/graphics/shaders/compile.bat \
    core/graphics/shaders/base/base.frag \
    core/graphics/shaders/base/base.vert \
    core/graphics/shaders/godRays/godRays.frag \
    core/graphics/shaders/godRays/godRays.vert \
    core/graphics/shaders/postProcessing/firstPostProcessingShader.frag \
    core/graphics/shaders/postProcessing/firstPostProcessingShader.vert \
    core/graphics/shaders/shadow/shadowMapShader.vert \
    core/graphics/shaders/postProcessing/postProcessingShader.frag \
    core/graphics/shaders/postProcessing/postProcessingShader.vert \
    core/graphics/shaders/bloomSprite/bloomSprite.frag \
    core/graphics/shaders/bloomSprite/bloomSprite.vert \
    core/graphics/shaders/skybox/skybox.frag \
    core/graphics/shaders/skybox/skybox.vert \
    model/glTF/Sponza/Sponza.gltf \
    model/glb/Bee.glb \
    model/glb/Box.glb \
    model/glb/Duck.glb \
    model/glb/RetroUFO.glb \
    texture/0.png \
    texture/1.png \
    texture/skybox/back.jpg \
    texture/skybox/bottom.jpg \
    texture/skybox/front.jpg \
    texture/skybox/left.jpg \
    texture/skybox/right.jpg \
    texture/skybox/top.jpg

SOURCES += \
    core/graphics/attachments.cpp \
    core/graphics/graphics.cpp \
    core/graphics/postProcessing.cpp \
    core/transformational/camera.cpp \
    core/transformational/group.cpp \
    core/transformational/light.cpp \
    core/transformational/object.cpp \
    core/transformational/gltfmodel.cpp \
    core/operations.cpp \
    core/texture.cpp \
    core/vulkanCore.cpp\
    core/control.cpp \
    main.cpp

HEADERS += \
    core/graphics/attachments.h \
    core/graphics/graphics.h \
    core/transformational/transformational.h \
    core/transformational/camera.h \
    core/transformational/group.h \
    core/transformational/light.h \
    core/transformational/object.h \
    core/transformational/gltfmodel.h \
    core/operations.h \
    core/texture.h \
    core/vulkanCore.h