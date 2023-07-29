TEMPLATE = lib
CONFIG += static
CONFIG += c++17 console
WARNINGS += -Wall

SOURCES += \
    matrix.cpp \
    quaternion.cpp \
    dualQuaternion.cpp \
    vector.cpp

HEADERS += \
    matrix.h \
    quaternion.h \
    dualQuaternion.h \
    vector.h

DISTFILES += \
    $$PWD/CMakelists.txt
