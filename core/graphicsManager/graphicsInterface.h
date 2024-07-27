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
    virtual utils::vkDefault::VkSemaphores submit(const uint32_t frameIndex, const utils::vkDefault::VkSemaphores& externalSemaphore) = 0;

    virtual void setProperties(
        const utils::PhysicalDeviceMap& devicesMap,
        const uint32_t deviceIndex,
        const utils::SwapChain* swapChain,
        uint32_t resources)
    {
        swapChainKHR = swapChain;
        resourceCount = resources;
        devices = &devicesMap;
        device = &devicesMap.at(deviceIndex);
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
