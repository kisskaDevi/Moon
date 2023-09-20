TEMPLATE = subdirs

SUBDIRS += \
    utils \
    graphicsManager \
    interfaces \
    deferredGraphics \
    imguiGraphics \
    models \
    transformational \
    math \
    testScene \
    testPos

utils.subdir = core/utils
graphicsManager.subdir = core/graphicsManager
interfaces.subdir = core/interfaces
deferredGraphics.subdir = core/deferredGraphics
imguiGraphics.subdir = core/imguiGraphics
models.subdir = core/models
transformational.subdir = core/transformational
math.subdir = core/math
testScene.subdir = tests/testScene
testPos.subdir = tests/testPos

graphicsManager.depends = utils
interfaces.depends = utils
models.depends = utils interfaces
transformational.depends = utils interfaces math
imguiGraphics.depends = utils graphicsManager
deferredGraphics.depends = utils interfaces graphicsManager
testScene.depends = graphicsManager imguiGraphics deferredGraphics models transformational
testPos.depends = graphicsManager imguiGraphics deferredGraphics models transformational

equals(QMAKE_CXX,cl){
    SUBDIRS += \
        cudaRayTracing \
        testCuda

    cudaRayTracing.subdir = core/cudaRayTracing
    cudaRayTracing.depends = utils graphicsManager
    testCuda.subdir = tests/testCuda
    testCuda.depends = graphicsManager cudaRayTracing
}

DISTFILES += \
    $$PWD/CMakelists.txt
