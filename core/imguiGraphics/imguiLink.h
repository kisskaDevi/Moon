#ifndef IMGUILINK_H
#define IMGUILINK_H

#include "linkable.h"

class imguiLink : public linkable
{
private:
    VkRenderPass renderPass{VK_NULL_HANDLE};

public:
    imguiLink() = default;

    void draw(VkCommandBuffer commandBuffer, uint32_t imageNumber) const override;
    void setRenderPass(VkRenderPass renderPass) override;

    const VkRenderPass& getRenderPass();
};
#endif // IMGUILINK_H
