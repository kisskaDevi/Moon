#ifndef GRAPHICSINTERFACE_H
#define GRAPHICSINTERFACE_H

#include <vulkan.h>
#include <vector>

#include "device.h"
#include "swapChain.h"

struct GLFWwindow;
class linkable;

class graphicsInterface{
protected:
    uint32_t                    imageCount{0};

    swapChain*                  swapChainKHR{nullptr};
    std::vector<physicalDevice> devices;
    physicalDevice              device;
    linkable*                   link{nullptr};

public:
    virtual ~graphicsInterface(){};
    virtual void destroyGraphics() = 0;

    virtual void setSwapChain(swapChain* swapChainKHR)
    {
        this->swapChainKHR = swapChainKHR;
        this->imageCount = swapChainKHR->getImageCount();
    }

    virtual void setDevices(uint32_t devicesCount, physicalDevice* devices)
    {
        for(uint32_t i=0;i<devicesCount;i++){
            this->devices.push_back(devices[i]);
        }
        device = this->devices.front();
    }
    virtual void createGraphics() = 0;

    virtual void updateCommandBuffer(uint32_t imageIndex) = 0;
    virtual void updateBuffers(uint32_t imageIndex) = 0;

    virtual std::vector<std::vector<VkSemaphore>> submit(
        const std::vector<std::vector<VkSemaphore>>& externalSemaphore,
        const std::vector<VkFence>& externalFence,
        uint32_t imageIndex) = 0;

    virtual linkable* getLinkable(){
        return link;
    }
};

#endif // GRAPHICSINTERFACE_H
