#ifndef GRAPHICSINTERFACE_H
#define GRAPHICSINTERFACE_H

#include <vulkan.h>
#include <vector>

struct GLFWwindow;
struct physicalDevice;
class swapChain;
class linkable;

class graphicsInterface{
public:
    virtual ~graphicsInterface(){};
    virtual void destroyGraphics() = 0;

    virtual void setDevices(uint32_t devicesCount, physicalDevice* devices) = 0;
    virtual void setSwapChain(swapChain* swapChainKHR) = 0;
    virtual void createGraphics() = 0;

    virtual void updateCommandBuffer(uint32_t imageIndex) = 0;
    virtual void updateBuffers(uint32_t imageIndex) = 0;

    virtual std::vector<std::vector<VkSemaphore>> submit(
        const std::vector<std::vector<VkSemaphore>>& externalSemaphore,
        const std::vector<VkFence>& externalFence,
        uint32_t imageIndex) = 0;

    virtual linkable* getLinkable() = 0;
};

#endif // GRAPHICSINTERFACE_H
