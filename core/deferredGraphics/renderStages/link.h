#ifndef LINK_H
#define LINK_H

#include <vector>
#include <filesystem>

#include "linkable.h"
#include "attachments.h"
#include "vector.h"

namespace moon::deferredGraphics {

struct LinkPushConstant{
    moon::math::Vector<float,2> offset{0.0f, 0.0f};
    moon::math::Vector<float,2> size{1.0f, 1.0f};
};

class Link : public moon::graphicsManager::Linkable
{
private:
    std::filesystem::path           shadersPath;
    uint32_t                        imageCount{0};
    VkDevice                        device{VK_NULL_HANDLE};
    VkRenderPass                    renderPass{VK_NULL_HANDLE};

    VkPipelineLayout                PipelineLayout{VK_NULL_HANDLE};
    VkPipeline                      Pipeline{VK_NULL_HANDLE};
    VkDescriptorSetLayout           DescriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorPool                DescriptorPool{VK_NULL_HANDLE};
    std::vector<VkDescriptorSet>    DescriptorSets;

    LinkPushConstant                pushConstant;

public:
    Link() = default;
    void destroy();
    void setDeviceProp(VkDevice device);
    void setImageCount(const uint32_t& count);
    void setShadersPath(const std::filesystem::path& shadersPath);
    void setPositionInWindow(const moon::math::Vector<float,2>& offset, const moon::math::Vector<float,2>& size);

    void draw(VkCommandBuffer commandBuffer, uint32_t imageNumber) const override;
    void setRenderPass(VkRenderPass renderPass) override;

    void createDescriptorSetLayout();
    void createPipeline(moon::utils::ImageInfo* pInfo);
    void createDescriptorPool();
    void createDescriptorSets();
    void updateDescriptorSets(const moon::utils::Attachments* attachment);
};

}
#endif // LINK_H
