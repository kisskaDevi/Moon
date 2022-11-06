CONFIG += c++17 console

win32: LIBS += -L$$PWD/libs/Lib/vulkan/x64/ -lvulkan-1

INCLUDEPATH += $$PWD/libs/Lib/vulkan/x64
DEPENDPATH += $$PWD/libs/Lib/vulkan/x64

win32: LIBS += -L$$PWD/libs/glfw-3.3.4.bin.WIN64/lib-static-ucrt/ -lglfw3dll

INCLUDEPATH += $$PWD/libs/glfw-3.3.4.bin.WIN64/lib-static-ucrt
DEPENDPATH += $$PWD/libs/glfw-3.3.4.bin.WIN64/lib-static-ucrt

DISTFILES += \
    core/graphics/deferredGraphics/shaders/SpotLightingPass/SpotLighting.frag \
    core/graphics/deferredGraphics/shaders/SpotLightingPass/SpotLighting.vert \
    core/graphics/deferredGraphics/shaders/SpotLightingPass/SpotLightingAmbient.frag \
    core/graphics/deferredGraphics/shaders/SpotLightingPass/SpotLightingAmbient.vert \
    core/graphics/deferredGraphics/shaders/SpotLightingPass/SpotLightingScattering.frag \
    core/graphics/deferredGraphics/shaders/combiner/combiner.frag \
    core/graphics/deferredGraphics/shaders/combiner/combiner.vert \
    core/graphics/deferredGraphics/shaders/compile.bat \
    core/graphics/deferredGraphics/shaders/base/base.frag \
    core/graphics/deferredGraphics/shaders/base/base.vert \
    core/graphics/deferredGraphics/shaders/compileBuild.bat \
    core/graphics/deferredGraphics/shaders/customFilter/customFilter.frag \
    core/graphics/deferredGraphics/shaders/customFilter/customFilter.vert \
    core/graphics/deferredGraphics/shaders/gaussianBlur/xBlur.frag \
    core/graphics/deferredGraphics/shaders/gaussianBlur/xBlur.vert \
    core/graphics/deferredGraphics/shaders/gaussianBlur/yBlur.frag \
    core/graphics/deferredGraphics/shaders/gaussianBlur/yBlur.vert \
    core/graphics/deferredGraphics/shaders/shadow/shadowMapShader.vert \
    core/graphics/deferredGraphics/shaders/postProcessing/postProcessingShader.frag \
    core/graphics/deferredGraphics/shaders/postProcessing/postProcessingShader.vert \
    core/graphics/deferredGraphics/shaders/skybox/skybox.frag \
    core/graphics/deferredGraphics/shaders/skybox/skybox.vert \
    core/graphics/deferredGraphics/shaders/ssao/SSAO.frag \
    core/graphics/deferredGraphics/shaders/ssao/SSAO.vert \
    core/graphics/deferredGraphics/shaders/sslr/SSLR.frag \
    core/graphics/deferredGraphics/shaders/sslr/SSLR.vert \
    core/graphics/deferredGraphics/shaders/stencil/secondStencil.frag \
    core/graphics/deferredGraphics/shaders/stencil/secondStencil.vert \
    model/glTF/Sponza/Sponza.gltf \
    model/glb/Bee.glb \
    model/glb/Box.glb \
    model/glb/Duck.glb \
    model/glb/RetroUFO.glb \
    model/glb/sponza.glb \
    texture/0.png \
    texture/1.png \
    texture/icon.ico \
    texture/skybox/back.jpg \
    texture/skybox/bottom.jpg \
    texture/skybox/front.jpg \
    texture/skybox/left.jpg \
    texture/skybox/right.jpg \
    texture/skybox/top.jpg

SOURCES += \
    core/graphics/deferredGraphics/attachments.cpp \
    core/graphics/deferredGraphics/deferredgraphicsinterface.cpp \
    core/graphics/deferredGraphics/filters/blur.cpp \
    core/graphics/deferredGraphics/renderStages/source/spotLighting.cpp \
    core/graphics/deferredGraphics/renderStages/source/base.cpp \
    core/graphics/deferredGraphics/renderStages/source/extension.cpp \
    core/graphics/deferredGraphics/renderStages/source/skybox.cpp \
    core/graphics/deferredGraphics/renderStages/graphics.cpp \
    core/graphics/deferredGraphics/renderStages/postProcessing.cpp \
    core/graphics/deferredGraphics/renderStages/shadowGraphics.cpp \
    core/graphics/deferredGraphics/filters/combiner.cpp \
    core/graphics/deferredGraphics/filters/customfilter.cpp \
    core/graphics/deferredGraphics/filters/ssao.cpp \
    core/graphics/deferredGraphics/filters/sslr.cpp \
    core/transformational/camera.cpp \
    core/transformational/group.cpp \
    core/transformational/light.cpp \
    core/transformational/object.cpp \
    core/transformational/gltfmodel.cpp \
    core/operations.cpp \
    core/texture.cpp \
    core/graphicsManager.cpp\
    physicalobject.cpp \
    main.cpp \
    scene.cpp \
    scene2.cpp

HEADERS += \
    core/graphics/deferredGraphics/attachments.h \
    core/graphics/deferredGraphics/bufferObjects.h \
    core/graphics/deferredGraphics/deferredgraphicsinterface.h \
    core/graphics/deferredGraphics/filters/blur.h \
    core/graphics/deferredGraphics/renderStages/graphics.h \
    core/graphics/deferredGraphics/renderStages/postProcessing.h \
    core/graphics/deferredGraphics/renderStages/shadowGraphics.h \
    core/graphics/deferredGraphics/filters/combiner.h \
    core/graphics/deferredGraphics/filters/customfilter.h \
    core/graphics/deferredGraphics/filters/ssao.h \
    core/graphics/deferredGraphics/filters/sslr.h \
    core/graphics/graphicsInterface.h \
    core/transformational/transformational.h \
    core/transformational/camera.h \
    core/transformational/group.h \
    core/transformational/light.h \
    core/transformational/object.h \
    core/transformational/gltfmodel.h \
    libs/dualQuaternion.h \
    libs/quaternion.h \
    physicalobject.h \
    core/operations.h \
    core/texture.h \
    core/graphicsManager.h \
    scene.h \
    scene2.h

win32:RC_ICONS += texture/icon.ico
