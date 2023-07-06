TEMPLATE = lib
CONFIG += static
CONFIG += c++17 console
WARNINGS += -Wall

INCLUDEPATH += \
    $$PWD/../../dependences/libs/glm \

HEADERS += \
    quaternion.h \
    dualQuaternion.h

DISTFILES += \
    $$PWD/CMakelists.txt
