TEMPLATE = app
CONFIG += c++17 console
WARNINGS += -Wall

INCLUDEPATH += \
    $$PWD/../../dependences\libs \
    $$PWD/../../dependences/libs/vulkan/include/vulkan \
    $$PWD/../../dependences/libs/glfw/include/GLFW \
    $$PWD/../../dependences/libs/glfw/include \
    $$PWD/../../dependences/libs/stb \
    $$PWD/../../core/cudaRayTracing/hitable \
    $$PWD/../../core/cudaRayTracing/interfaces \
    $$PWD/../../core/cudaRayTracing/materials \
    $$PWD/../../core/cudaRayTracing/math \
    $$PWD/../../core/cudaRayTracing/rayTracingGraphics \
    $$PWD/../../core/cudaRayTracing/transformational \
    $$PWD/../../core/cudaRayTracing/utils \
    $$PWD/../../core/graphicsManager \
    $$PWD/../../core/utils \
    $$PWD/../.. \
    $$PWD/..

equals(QMAKE_CXX,cl){
    DEFINES += TESTCUDA

    Release:DESTDIR = release
    Debug:DESTDIR = debug

    CUDA_DIR = $$(CUDA_PATH)
    CUDA_INCLUDE_DIR = $$CUDA_DIR/include
    CUDA_LIBS_DIR = $$CUDA_DIR/lib/x64

    Release:GLFW_BIN_DIR = Release
    Debug:GLFW_BIN_DIR = Debug

    win32: LIBS += \
        -L$$OUT_PWD/../../core/graphicsManager/$$DESTDIR \
        -L$$OUT_PWD/../../core/utils/$$DESTDIR \
        -L$$PWD/../../dependences/libs/vulkan_runtime/x64 \
        -L$$PWD/../../dependences/libs/glfw/build/src/$$GLFW_BIN_DIR \
        -L$$OUT_PWD/../../core/cudaRayTracing/$$DESTDIR \
        -L$$CUDA_LIBS_DIR \
        -lgraphicsManager \
        -lcudaRayTracing \
        -lutils \
        -lvulkan-1 \
        -lglfw3dll \
        -lcuda -lcudart -lcudadevrt -lcudart_static

    INCLUDEPATH += \
        $$CUDA_INCLUDE_DIR\

    HEADERS += \
        ../scene.h \
        ../controller.h \
        testCuda.h
}
SOURCES += \
    ../main.cpp \
    ../controller.cpp \
    testCuda.cpp

DISTFILES += \
    $$PWD/CMakelists.txt
