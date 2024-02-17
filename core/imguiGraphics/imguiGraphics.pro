TEMPLATE = lib
CONFIG += static
CONFIG += c++17 console
WARNINGS += -Wall

INCLUDEPATH += \
    $$PWD/../../dependences/libs/vulkan/include/vulkan \
    $$PWD/../../dependences/libs/glfw/include/GLFW \
    $$PWD/../../dependences/libs/glfw/include \
    $$PWD/../utils \
    $$PWD/../math \
    $$PWD/../graphicsManager \
    $$PWD/../../dependences/libs/imgui \
    $$PWD/../../dependences/libs/imgui/backends

SOURCES += \
    imguiGraphics.cpp \
    $$PWD/../../dependences/libs/imgui/imgui.cpp \
    $$PWD/../../dependences/libs/imgui/imgui_draw.cpp \
    $$PWD/../../dependences/libs/imgui/imgui_tables.cpp \
    $$PWD/../../dependences/libs/imgui/imgui_widgets.cpp \
    $$PWD/../../dependences/libs/imgui/imgui_demo.cpp \
    $$PWD/../../dependences/libs/imgui/backends/imgui_impl_vulkan.cpp \
    $$PWD/../../dependences/libs/imgui/backends/imgui_impl_glfw.cpp \
    imguiLink.cpp

HEADERS += \
    imguiGraphics.h \
    $$PWD/../../dependences/libs/imgui/imgui.h \
    $$PWD/../../dependences/libs/imgui/backends/imgui_impl_vulkan.h \
    $$PWD/../../dependences/libs/imgui/backends/imgui_impl_glfw.h \
    imguiLink.h

DISTFILES += \
    $$PWD/CMakelists.txt
