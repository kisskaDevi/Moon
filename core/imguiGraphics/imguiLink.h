#ifndef IMGUILINK_H
#define IMGUILINK_H

#include <vector>
#include <filesystem>

#include "linkable.h"

class imguiLink : public linkable
{
private:
    uint32_t                        imageCount{0};
    VkDevice                        device{VK_NULL_HANDLE};
    VkRenderPass                    renderPass{VK_NULL_HANDLE};

public:
    imguiLink() = default;
    void setDeviceProp(VkDevice device);
    void setImageCount(const uint32_t& count);

    void draw(VkCommandBuffer commandBuffer, uint32_t imageNumber) const override;
    void setRenderPass(VkRenderPass renderPass) override;

    const VkRenderPass& getRenderPass();
};
#endif // IMGUILINK_H
