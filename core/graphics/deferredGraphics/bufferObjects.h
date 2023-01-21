#ifndef BUFFEROBJECTS_H
#define BUFFEROBJECTS_H

#include <libs/glm/glm.hpp>

struct StorageBufferObject{
    alignas(16) glm::vec4           mousePosition;
    alignas(4)  int                 number;
    alignas(4)  float               depth;
};

#endif // BUFFEROBJECTS_H
