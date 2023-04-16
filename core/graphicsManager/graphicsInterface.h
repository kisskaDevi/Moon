#ifndef GRAPHICSINTERFACE_H
#define GRAPHICSINTERFACE_H

#include <vulkan.h>
#include <vector>

class GLFWwindow;
class physicalDevice;

class graphicsInterface{
public:
    virtual ~graphicsInterface(){};
    virtual void destroyGraphics() = 0;
    virtual void destroyCommandPool() = 0;

    virtual void setDevices(uint32_t devicesCount, physicalDevice* devices) = 0;
    virtual void setSupportImageCount(VkSurfaceKHR* surface) = 0;
    virtual void createCommandPool() = 0;
    virtual void createGraphics(GLFWwindow* window, VkSurfaceKHR* surface) = 0;
    virtual void updateDescriptorSets() = 0;

    virtual void createCommandBuffers() = 0;
    virtual void updateCommandBuffers() = 0;
    virtual void updateCommandBuffer(uint32_t imageIndex) = 0;
    virtual void freeCommandBuffers() = 0;

    virtual void updateBuffers(uint32_t imageIndex) = 0;

    virtual std::vector<std::vector<VkSemaphore>> sibmit(std::vector<std::vector<VkSemaphore>>& externalSemaphore, std::vector<VkFence>& externalFence, uint32_t imageIndex) = 0;

    virtual uint32_t            getImageCount() = 0;
    virtual VkSwapchainKHR&     getSwapChain() = 0;
};

#endif // GRAPHICSINTERFACE_H
