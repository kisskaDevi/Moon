#ifndef CAMERA_H
#define CAMERA_H

#include <vulkan.h>

class camera{
public:
    virtual void destroy(VkDevice* device) = 0;

    virtual VkBuffer getBuffer(uint32_t index) const = 0;
    virtual VkDeviceSize getBufferRange() const = 0;

    virtual void createUniformBuffers(VkPhysicalDevice* physicalDevice, VkDevice* device, uint32_t imageCount) = 0;
    virtual void updateUniformBuffer(VkCommandBuffer commandBuffer, uint32_t frameNumber) = 0;
};

#endif // CAMERA_H
