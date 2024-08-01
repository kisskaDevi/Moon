#ifndef LINKABLE_H
#define LINKABLE_H

#include <vulkan.h>
#include "vector.h"

namespace moon::graphicsManager {

struct PositionInWindow {
    moon::math::Vector<float, 2> offset{ 0.0f, 0.0f };
    moon::math::Vector<float, 2> size{ 1.0f, 1.0f };
};

class Linkable{
protected:
    VkRenderPass pRenderPass{ VK_NULL_HANDLE };
    PositionInWindow positionInWindow;

public:
    Linkable() = default;
    Linkable(VkRenderPass renderPass) : pRenderPass(renderPass) {}
    virtual ~Linkable(){};
    virtual void draw(VkCommandBuffer commandBuffer, uint32_t imageNumber) const = 0;
    virtual void setPositionInWindow(const PositionInWindow& position){ positionInWindow = position; }
    virtual VkRenderPass& renderPass() { return pRenderPass; }
};

}
#endif // LINKABLE_H
