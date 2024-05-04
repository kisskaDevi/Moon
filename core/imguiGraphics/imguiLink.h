#ifndef IMGUILINK_H
#define IMGUILINK_H

#include "linkable.h"

namespace moon::imguiGraphics {

class ImguiLink : public moon::graphicsManager::Linkable
{
private:
    VkRenderPass renderPass{VK_NULL_HANDLE};

public:
    ImguiLink() = default;

    void draw(VkCommandBuffer commandBuffer, uint32_t imageNumber) const override;
    void setRenderPass(VkRenderPass renderPass) override;

    const VkRenderPass& getRenderPass();
};

}
#endif // IMGUILINK_H
