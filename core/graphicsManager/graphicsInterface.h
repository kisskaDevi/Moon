#ifndef GRAPHICSINTERFACE_H
#define GRAPHICSINTERFACE_H

#include <vector>
#include <memory>

#include <vulkan.h>
#include "device.h"
#include "vector.h"
#include "linkable.h"
#include "swapChain.h"

namespace moon::graphicsManager {

class GraphicsManager;
class Linkable;

class GraphicsInterface{
protected:
    const utils::PhysicalDeviceMap* devices{ nullptr };
    const utils::PhysicalDevice*    device{ nullptr };
    const utils::SwapChain*         swapChainKHR{ nullptr };

    uint32_t                        resourceCount{ 0 };
    math::Vector<float,2>           offset{0.0f, 0.0f};
    math::Vector<float,2>           size{1.0f, 1.0f};
    std::unique_ptr<Linkable>       link;

private:
    virtual void update(uint32_t imageIndex) = 0;

    virtual std::vector<std::vector<VkSemaphore>> submit(
        const std::vector<std::vector<VkSemaphore>>& externalSemaphore,
        const std::vector<VkFence>& externalFence,
        uint32_t imageIndex) = 0;

    virtual void setProperties(
        const utils::PhysicalDeviceMap& devices,
        const uint32_t deviceIndex,
        const utils::SwapChain* swapChain,
        uint32_t resourceCount)
    {
        this->swapChainKHR = swapChain;
        this->resourceCount = resourceCount;
        this->devices = &devices;
        device = &devices.at(deviceIndex);
    }

    friend class GraphicsLinker;
    friend class GraphicsManager;

public:
    virtual ~GraphicsInterface(){};

    virtual void reset() = 0;

    virtual void setPositionInWindow(const math::Vector<float,2>& offset, const math::Vector<float,2>& size) {
        if(link) link->setPositionInWindow({ offset , size });
    }
};

}
#endif // GRAPHICSINTERFACE_H
