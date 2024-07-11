#ifndef GRAPHICSINTERFACE_H
#define GRAPHICSINTERFACE_H

#include <vulkan.h>
#include <vector>

#include "device.h"
#include "vector.h"

namespace moon::utils { class SwapChain;}

namespace moon::graphicsManager {

class Linkable;

class GraphicsInterface{
protected:
    VkFormat                                         format{VK_FORMAT_UNDEFINED};
    uint32_t                                         imageCount{0};
    const std::map<uint32_t, moon::utils::PhysicalDevice>* devices;
    const moon::utils::PhysicalDevice*                     device;
    moon::utils::SwapChain*                          swapChainKHR;
    Linkable*                                        link{nullptr};
    moon::math::Vector<float,2>                      offset{0.0f, 0.0f};
    moon::math::Vector<float,2>                      size{1.0f, 1.0f};

public:
    virtual ~GraphicsInterface(){};

    virtual void setPositionInWindow(const moon::math::Vector<float,2>& offset, const moon::math::Vector<float,2>& size){
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
        this->devices = &devices;
        device = &devices.at(deviceIndex);
    }

    virtual Linkable* getLinkable(){
        return link;
    }

    virtual void reset() = 0;
    virtual void update(uint32_t imageIndex) = 0;

    virtual std::vector<std::vector<VkSemaphore>> submit(
        const std::vector<std::vector<VkSemaphore>>& externalSemaphore,
        const std::vector<VkFence>& externalFence,
        uint32_t imageIndex) = 0;
};

}
#endif // GRAPHICSINTERFACE_H
