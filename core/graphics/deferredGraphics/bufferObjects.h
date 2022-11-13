#ifndef BUFFEROBJECTS_H
#define BUFFEROBJECTS_H

#include <libs/glm/glm.hpp>

struct UniformBufferObject{
    alignas(16) glm::mat4           view;
    alignas(16) glm::mat4           proj;
    alignas(16) glm::vec4           eyePosition;
    alignas(4)  float               enableTransparency;
};

struct SkyboxUniformBufferObject{
    alignas(16) glm::mat4           proj;
    alignas(16) glm::mat4           view;
    alignas(16) glm::mat4           model;
};

struct StorageBufferObject{
    alignas(16) glm::vec4           mousePosition;
    alignas(4)  int                 number;
    alignas(4)  float               depth;
};

struct StencilPushConst{
    alignas(16) glm::vec4           stencilColor;
    alignas(4)  float               width;
};

struct lightPassPushConst{
    alignas(4) float                minAmbientFactor;
};

struct postProcessingPushConst{
    alignas(4) float                blitFactor;
};

struct CustomFilterPushConst{
    alignas (4) float deltax;
    alignas (4) float deltay;
};

#endif // BUFFEROBJECTS_H
