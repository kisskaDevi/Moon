#ifndef CAMERA_H
#define CAMERA_H

#include <vulkan.h>
#include "buffer.h"

struct physicalDevice;

class camera{
public:
    virtual ~camera(){};

    virtual void destroy(VkDevice device) = 0;

    virtual const buffers& getBuffers() const = 0;

    virtual void create(physicalDevice device, uint32_t imageCount) = 0;
    virtual void update(uint32_t frameNumber, VkCommandBuffer commandBuffer) = 0;
};

#endif // CAMERA_H
