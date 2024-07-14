#ifndef LINKABLE_H
#define LINKABLE_H

#include <vulkan.h>
#include "vector.h"

namespace moon::graphicsManager {

class Linkable{
protected:
    VkRenderPass pRenderPass{ VK_NULL_HANDLE };

public:
    virtual ~Linkable(){};
    virtual void setPositionInWindow(const math::Vector<float, 2>& offset, const math::Vector<float, 2>& size) = 0;
    virtual void draw(VkCommandBuffer commandBuffer, uint32_t imageNumber) const = 0;
    virtual VkRenderPass& renderPass() { return pRenderPass; }
};

}
#endif // LINKABLE_H
