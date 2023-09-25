TEMPLATE = subdirs

SUBDIRS += \
    utils \
    workflows \
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
workflows.subdir = core/workflows
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
models.depends = utils interfaces math
transformational.depends = utils interfaces math
imguiGraphics.depends = utils graphicsManager
workflows.depends = utils interfaces math
deferredGraphics.depends = utils interfaces workflows graphicsManager math
testScene.depends = graphicsManager imguiGraphics deferredGraphics models transformational math
testPos.depends = graphicsManager imguiGraphics deferredGraphics models transformational math

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
