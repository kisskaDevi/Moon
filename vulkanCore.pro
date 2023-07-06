TEMPLATE = subdirs

SUBDIRS += \
    utils \
    graphicsManager \
    interfaces \
    deferredGraphics \
    models \
    transformational \
    test \
    math

utils.subdir = core/utils
graphicsManager.subdir = core/graphicsManager
interfaces.subdir = core/interfaces
deferredGraphics.subdir = core/deferredGraphics
models.subdir = core/models
transformational.subdir = core/transformational
math.subdir = core/math
test.subdir = test

graphicsManager.depends = utils
interfaces.depends = utils
deferredGraphics.depends = utils interfaces graphicsManager
models.depends = utils interfaces
transformational.depends = utils interfaces math
test.depends = graphicsManager deferredGraphics models transformational

DISTFILES += \
    $$PWD/CMakelists.txt
