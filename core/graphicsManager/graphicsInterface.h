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
    virtual void destroy() = 0;

    virtual void setSwapChain(swapChain* swapChainKHR){
        this->swapChainKHR = swapChainKHR;
        this->imageCount = swapChainKHR->getImageCount();
    }

    virtual void setDevices(uint32_t devicesCount, physicalDevice* devices, int deviceIndex = -1){
        for(uint32_t i = 0; i < devicesCount; i++){
            this->devices.push_back(devices[i]);
        }
        device = deviceIndex == -1 ? this->devices.front() : this->devices[deviceIndex];
    }

    virtual linkable* getLinkable(){
        return link;
    }

    virtual void create() = 0;

    virtual void update(uint32_t imageIndex) = 0;

    virtual std::vector<std::vector<VkSemaphore>> submit(
        const std::vector<std::vector<VkSemaphore>>& externalSemaphore,
        const std::vector<VkFence>& externalFence,
        uint32_t imageIndex) = 0;
};

#endif // GRAPHICSINTERFACE_H
