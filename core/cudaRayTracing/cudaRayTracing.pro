TEMPLATE = lib
CONFIG += static
CONFIG += c++17 console
WARNINGS += -Wall

INCLUDEPATH += \
    $$PWD/../../dependences/libs/vulkan/include/vulkan \
    $$PWD/../graphicsManager \
    $$PWD/../..

HEADERS += \
    $$PWD/hitable/sphere.h \
    $$PWD/hitable/triangle.h \
    $$PWD/hitable/hitable.h \
    $$PWD/materials/material.h \
    $$PWD/math/dualQuaternion.h \
    $$PWD/math/quaternion.h \
    $$PWD/math/ray.h \
    $$PWD/math/vec4.h \
    $$PWD/rayTracingGraphics/rayTracingGraphics.h \
    $$PWD/rayTracingGraphics/rayTracingLink.h \
    $$PWD/transformational/camera.h \
    $$PWD/transformational/object.h \
    $$PWD/utils/buffer.h \
    $$PWD/utils/hitableList.h \
    $$PWD/utils/hitableContainer.h \
    $$PWD/utils/hitableArray.h \
    $$PWD/utils/operations.h \
    utils/hitableArray.h \
    utils/hitableContainer.h

SOURCES += \
    $$PWD/rayTracingGraphics/rayTracingLink.cpp

CUDA_SOURCES += \
    $$PWD/hitable/sphere.cu \
    $$PWD/hitable/triangle.cu \
    $$PWD/math/vec4.cu \
    $$PWD/rayTracingGraphics/rayTracingGraphics.cu \
    $$PWD/transformational/camera.cu \
    $$PWD/utils/hitableList.cu \
    $$PWD/utils/hitableArray.cu \
    $$PWD/utils/hitableContainer.cu \
    $$PWD/utils/operations.cu \

CUDA_ARCH = compute_86
CUDA_SM = sm_86
SYSTEM_TYPE = 64
Release:MSVCRT_LINK_FLAG = "/MD"
Debug:MSVCRT_LINK_FLAG = "/MDd"
XFLAGS = -Xcompiler "/EHsc,/W3,/nologo,/O2,/FS,$$MSVCRT_LINK_FLAG"

CUDA_DIR = $$(CUDA_PATH)
CUDA_INCLUDE_DIR = $$CUDA_DIR/include
CUDA_LIBS_DIR = $$CUDA_DIR/lib/x64

CUDA_INCLUDEPATH += \
    -I$$CUDA_INCLUDE_DIR \
    -I$$PWD/hitable \
    -I$$PWD/interfaces \
    -I$$PWD/materials \
    -I$$PWD/math \
    -I$$PWD/rayTracingGraphics \
    -I$$PWD/transformational \
    -I$$PWD/utils \
    -I$$PWD/../../dependences/libs/vulkan/include/vulkan \
    -I$$PWD/../graphicsManager \
    -I$$PWD/../..

QMAKE_LIBDIR += -L$$CUDA_LIBS_DIR
LIBS += $$QMAKE_LIBDIR -lcuda -lcudart -lcudadevrt -lcudart_static

Release: config = release
Debug: config = debug

cuda.input = CUDA_SOURCES
cuda.output =  $$OUT_PWD/$$config/cuda_obj/${QMAKE_FILE_BASE}.obj
cuda.commands =         $${CUDA_DIR}/bin/nvcc.exe \
                        -gencode arch=$$CUDA_ARCH,code=$$CUDA_SM \
                        -x cu \
                        -rdc=true \
                        -std=c++17 \
                        $$CUDA_INCLUDEPATH \
                        --machine $$SYSTEM_TYPE \
                        --compile -cudart static \
                        -DNDEBUG -D_CONSOLE -D_UNICODE -DUNICODE \
                        $$XFLAGS \
                        -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
QMAKE_EXTRA_COMPILERS += cuda

CUDA_OBJS += \
    $$OUT_PWD/$$config/cuda_obj/sphere.obj \
    $$OUT_PWD/$$config/cuda_obj/triangle.obj \
    $$OUT_PWD/$$config/cuda_obj/vec4.obj \
    $$OUT_PWD/$$config/cuda_obj/rayTracingGraphics.obj \
    $$OUT_PWD/$$config/cuda_obj/camera.obj \
    $$OUT_PWD/$$config/cuda_obj/hitableList.obj \
    $$OUT_PWD/$$config/cuda_obj/hitableContainer.obj \
    $$OUT_PWD/$$config/cuda_obj/hitableArray.obj \
    $$OUT_PWD/$$config/cuda_obj/operations.obj \

cuda_l.input =      CUDA_OBJS
cuda_l.output =     $$OUT_PWD/$$config/cuda_obj/device_link.o
cuda_l.commands =   $$CUDA_DIR/bin/nvcc \
                    -gencode arch=$$CUDA_ARCH,code=$$CUDA_SM \
                    $$NVCCFLAGS \
                    -std=c++17 \
                    -dlink \
                    -o ${QMAKE_FILE_OUT} \
                    $$LIBS $$CUDA_OBJS
cuda_l.dependency_type = TYPE_C
cuda_l.clean = $$OUT_PWD/$$config/cuda_obj/device_link.o
QMAKE_EXTRA_UNIX_COMPILERS += cuda_l

DISTFILES += \
    $$PWD/CMakelists.txt

