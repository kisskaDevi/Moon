#ifndef GRAPHICSINTERFACE_H
#define GRAPHICSINTERFACE_H

#include <vulkan.h>
#include <vector>

#include "device.h"
#include "vector.h"

namespace moon::utils { class SwapChain;}

namespace moon::graphicsManager {

class Linkable;
class GraphicsManager;

class GraphicsInterface{
protected:
    uint32_t                        resourceCount{0};
    const utils::PhysicalDeviceMap* devices{ nullptr };
    const utils::PhysicalDevice*    device{ nullptr };
    utils::SwapChain*               swapChainKHR{ nullptr };
    Linkable*                       link{ nullptr };
    math::Vector<float,2>           offset{0.0f, 0.0f};
    math::Vector<float,2>           size{1.0f, 1.0f};

private:
    virtual void update(uint32_t imageIndex) = 0;

    virtual std::vector<std::vector<VkSemaphore>> submit(
        const std::vector<std::vector<VkSemaphore>>& externalSemaphore,
        const std::vector<VkFence>& externalFence,
        uint32_t imageIndex) = 0;

    virtual void setProperties(
        const utils::PhysicalDeviceMap& devices,
        const uint32_t deviceIndex,
        utils::SwapChain* swapChain,
        uint32_t resourceCount)
    {
        this->swapChainKHR = swapChain;
        this->resourceCount = resourceCount;
        this->devices = &devices;
        device = &devices.at(deviceIndex);
    }

    virtual Linkable* linkable() {
        return link;
    }

    friend class GraphicsManager;

public:
    virtual ~GraphicsInterface(){};

    virtual void reset() = 0;

    virtual void setPositionInWindow(const math::Vector<float,2>& offset, const math::Vector<float,2>& size) {
        this->offset = offset;
        this->size = size;
    }
};

}
#endif // GRAPHICSINTERFACE_H
