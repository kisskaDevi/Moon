TEMPLATE = subdirs

SUBDIRS += \
    utils \
    graphicsManager \
    interfaces \
    deferredGraphics \
    imguiGraphics \
    models \
    transformational \
    test \
    math

utils.subdir = core/utils
graphicsManager.subdir = core/graphicsManager
interfaces.subdir = core/interfaces
deferredGraphics.subdir = core/deferredGraphics
imguiGraphics.subdir = core/imguiGraphics
models.subdir = core/models
transformational.subdir = core/transformational
math.subdir = core/math
test.subdir = test

graphicsManager.depends = utils
interfaces.depends = utils
models.depends = utils interfaces
transformational.depends = utils interfaces math
imguiGraphics.depends = utils graphicsManager
deferredGraphics.depends = utils interfaces graphicsManager
test.depends = graphicsManager imguiGraphics deferredGraphics models transformational

DISTFILES += \
    $$PWD/CMakelists.txt
