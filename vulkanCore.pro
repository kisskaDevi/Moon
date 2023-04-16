TEMPLATE = subdirs

SUBDIRS += \
    utils \
    graphicsManager \
    interfaces \
    deferredGraphics \
    models \
    transformational \
    test

utils.subdir = core/utils
graphicsManager.subdir = core/graphicsManager
interfaces.subdir = core/interfaces
deferredGraphics.subdir = core/deferredGraphics
models.subdir = core/models
transformational.subdir = core/transformational
test.subdir = test

graphicsManager.depends = utils
interfaces.depends = utils
deferredGraphics.depends = utils interfaces graphicsManager
models.depends = utils interfaces
transformational.depends = utils interfaces
test.depends = graphicsManager deferredGraphics models transformational
