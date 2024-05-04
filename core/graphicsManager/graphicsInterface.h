#ifndef GRAPHICSINTERFACE_H
#define GRAPHICSINTERFACE_H

#include <vulkan.h>
#include <vector>

#include "device.h"
#include "vector.h"

class linkable;
namespace moon::utils { class SwapChain;}

class graphicsInterface{
protected:
    VkFormat                                         format{VK_FORMAT_UNDEFINED};
    uint32_t                                         imageCount{0};
    std::map<uint32_t, moon::utils::PhysicalDevice>  devices;
    moon::utils::PhysicalDevice                      device;
    moon::utils::SwapChain*                          swapChainKHR;
    linkable*                                        link{nullptr};
    vector<float,2>                                  offset{0.0f, 0.0f};
    vector<float,2>                                  size{1.0f, 1.0f};

public:
    virtual ~graphicsInterface(){};
    virtual void destroy() = 0;

    virtual void setPositionInWindow(const vector<float,2>& offset, const vector<float,2>& size){
        this->offset = offset;
        this->size = size;
    }

    virtual void setSwapChain(moon::utils::SwapChain* swapChain){
        this->swapChainKHR = swapChain;
    }

    virtual void setProperties(VkFormat swapChainFormat, uint32_t resourceCount){
        format = swapChainFormat;
        imageCount = resourceCount;
    }

    virtual void setDevices(const std::map<uint32_t, moon::utils::PhysicalDevice>& devices, uint32_t deviceIndex = 0xffffffff){
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
