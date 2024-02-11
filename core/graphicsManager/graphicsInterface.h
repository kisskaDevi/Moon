#ifndef GRAPHICSINTERFACE_H
#define GRAPHICSINTERFACE_H

#include <vulkan.h>
#include <vector>

#include "device.h"

struct GLFWwindow;
class linkable;

class graphicsInterface{
protected:
    VkFormat                            format{VK_FORMAT_UNDEFINED};
    uint32_t                            imageCount{0};
    std::map<uint32_t, physicalDevice>  devices;
    physicalDevice                      device;
    linkable*                           link{nullptr};

public:
    virtual ~graphicsInterface(){};
    virtual void destroy() = 0;

    virtual void setProperties(VkFormat swapChainFormat, uint32_t resourceCount){
        format = swapChainFormat;
        imageCount = resourceCount;
    }

    virtual void setDevices(const std::map<uint32_t, physicalDevice>& devices, uint32_t deviceIndex = 0xffffffff){
        this->devices = devices;
        device = devices.at(deviceIndex);
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
