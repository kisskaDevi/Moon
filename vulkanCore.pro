CONFIG += c++17 console

win32: LIBS += -L$$PWD/libs/Lib/vulkan/x64/ -lvulkan-1

INCLUDEPATH += $$PWD/libs/Lib/vulkan/x64
DEPENDPATH += $$PWD/libs/Lib/vulkan/x64

win32: LIBS += -L$$PWD/libs/glfw-3.3.4.bin.WIN64/lib-static-ucrt/ -lglfw3dll

INCLUDEPATH += $$PWD/libs/glfw-3.3.4.bin.WIN64/lib-static-ucrt
DEPENDPATH += $$PWD/libs/glfw-3.3.4.bin.WIN64/lib-static-ucrt

DISTFILES += \
    core/graphics/shaders/base/basefrag.spv \
    core/graphics/shaders/base/basevert.spv \
    core/graphics/shaders/bloomSprite/fragBloomSprite.spv \
    core/graphics/shaders/bloomSprite/lightBoxfrag.spv \
    core/graphics/shaders/bloomSprite/lightBoxvert.spv \
    core/graphics/shaders/bloomSprite/vertBloomSprite.spv \
    core/graphics/shaders/compile.bat \
    core/graphics/shaders/base/base.frag \
    core/graphics/shaders/base/base.vert \
    core/graphics/shaders/godRays/godRays.frag \
    core/graphics/shaders/godRays/godRays.vert \
    core/graphics/shaders/godRays/godRaysFrag.spv \
    core/graphics/shaders/godRays/godRaysVert.spv \
    core/graphics/shaders/postProcessing/postProcessingFrag.spv \
    core/graphics/shaders/postProcessing/postProcessingVert.spv \
    core/graphics/shaders/shadow/shad.spv \
    core/graphics/shaders/shadow/shadowMapShader.vert \
    core/graphics/shaders/postProcessing/postProcessingShader.frag \
    core/graphics/shaders/postProcessing/postProcessingShader.vert \
    core/graphics/shaders/bloomSprite/bloomSprite.frag \
    core/graphics/shaders/bloomSprite/bloomSprite.vert \
    core/graphics/shaders/skybox/skybox.frag \
    core/graphics/shaders/skybox/skybox.vert \
    core/graphics/shaders/skybox/skyboxFrag.spv \
    core/graphics/shaders/skybox/skyboxVert.spv \
    model/glTF/Sponza/10381718147657362067.jpg \
    model/glTF/Sponza/10388182081421875623.jpg \
    model/glTF/Sponza/11474523244911310074.jpg \
    model/glTF/Sponza/11490520546946913238.jpg \
    model/glTF/Sponza/11872827283454512094.jpg \
    model/glTF/Sponza/11968150294050148237.jpg \
    model/glTF/Sponza/1219024358953944284.jpg \
    model/glTF/Sponza/12501374198249454378.jpg \
    model/glTF/Sponza/13196865903111448057.jpg \
    model/glTF/Sponza/13824894030729245199.jpg \
    model/glTF/Sponza/13982482287905699490.jpg \
    model/glTF/Sponza/14118779221266351425.jpg \
    model/glTF/Sponza/14170708867020035030.jpg \
    model/glTF/Sponza/14267839433702832875.jpg \
    model/glTF/Sponza/14650633544276105767.jpg \
    model/glTF/Sponza/15295713303328085182.jpg \
    model/glTF/Sponza/15722799267630235092.jpg \
    model/glTF/Sponza/16275776544635328252.png \
    model/glTF/Sponza/16299174074766089871.jpg \
    model/glTF/Sponza/16885566240357350108.jpg \
    model/glTF/Sponza/17556969131407844942.jpg \
    model/glTF/Sponza/17876391417123941155.jpg \
    model/glTF/Sponza/2051777328469649772.jpg \
    model/glTF/Sponza/2185409758123873465.jpg \
    model/glTF/Sponza/2299742237651021498.jpg \
    model/glTF/Sponza/2374361008830720677.jpg \
    model/glTF/Sponza/2411100444841994089.jpg \
    model/glTF/Sponza/2775690330959970771.jpg \
    model/glTF/Sponza/2969916736137545357.jpg \
    model/glTF/Sponza/332936164838540657.jpg \
    model/glTF/Sponza/3371964815757888145.jpg \
    model/glTF/Sponza/3455394979645218238.jpg \
    model/glTF/Sponza/3628158980083700836.jpg \
    model/glTF/Sponza/3827035219084910048.jpg \
    model/glTF/Sponza/4477655471536070370.jpg \
    model/glTF/Sponza/4601176305987539675.jpg \
    model/glTF/Sponza/466164707995436622.jpg \
    model/glTF/Sponza/4675343432951571524.jpg \
    model/glTF/Sponza/4871783166746854860.jpg \
    model/glTF/Sponza/4910669866631290573.jpg \
    model/glTF/Sponza/4975155472559461469.jpg \
    model/glTF/Sponza/5061699253647017043.png \
    model/glTF/Sponza/5792855332885324923.jpg \
    model/glTF/Sponza/5823059166183034438.jpg \
    model/glTF/Sponza/6047387724914829168.jpg \
    model/glTF/Sponza/6151467286084645207.jpg \
    model/glTF/Sponza/6593109234861095314.jpg \
    model/glTF/Sponza/6667038893015345571.jpg \
    model/glTF/Sponza/6772804448157695701.jpg \
    model/glTF/Sponza/7056944414013900257.jpg \
    model/glTF/Sponza/715093869573992647.jpg \
    model/glTF/Sponza/7268504077753552595.jpg \
    model/glTF/Sponza/7441062115984513793.jpg \
    model/glTF/Sponza/755318871556304029.jpg \
    model/glTF/Sponza/759203620573749278.jpg \
    model/glTF/Sponza/7645212358685992005.jpg \
    model/glTF/Sponza/7815564343179553343.jpg \
    model/glTF/Sponza/8006627369776289000.png \
    model/glTF/Sponza/8051790464816141987.jpg \
    model/glTF/Sponza/8114461559286000061.jpg \
    model/glTF/Sponza/8481240838833932244.jpg \
    model/glTF/Sponza/8503262930880235456.jpg \
    model/glTF/Sponza/8747919177698443163.jpg \
    model/glTF/Sponza/8750083169368950601.jpg \
    model/glTF/Sponza/8773302468495022225.jpg \
    model/glTF/Sponza/8783994986360286082.jpg \
    model/glTF/Sponza/9288698199695299068.jpg \
    model/glTF/Sponza/9916269861720640319.jpg \
    model/glTF/Sponza/Sponza.bin \
    model/glTF/Sponza/Sponza.gltf \
    model/glTF/Sponza/white.png \
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
