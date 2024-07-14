#ifndef IMGUILINK_H
#define IMGUILINK_H

#include "linkable.h"

namespace moon::imguiGraphics {

class ImguiLink : public moon::graphicsManager::Linkable
{
public:
    void draw(VkCommandBuffer commandBuffer, uint32_t imageNumber) const override;
    void setPositionInWindow(const math::Vector<float, 2>& offset, const math::Vector<float, 2>& size) override {}
};

}
#endif // IMGUILINK_H
