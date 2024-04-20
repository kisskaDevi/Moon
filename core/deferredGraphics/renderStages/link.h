#ifndef LINK_H
#define LINK_H

#include <vector>
#include <filesystem>

#include "linkable.h"
#include "attachments.h"
#include "vector.h"

struct linkPushConstant{
    vector<float,2> offset{0.0f, 0.0f};
    vector<float,2> size{1.0f, 1.0f};
};

class link : public linkable
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

    linkPushConstant                pushConstant;

public:
    link() = default;
    void destroy();
    void setDeviceProp(VkDevice device);
    void setImageCount(const uint32_t& count);
    void setShadersPath(const std::filesystem::path& shadersPath);
    void setPositionInWindow(const vector<float,2>& offset, const vector<float,2>& size);

    void draw(VkCommandBuffer commandBuffer, uint32_t imageNumber) const override;
    void setRenderPass(VkRenderPass renderPass) override;

    void createDescriptorSetLayout();
    void createPipeline(imageInfo* pInfo);
    void createDescriptorPool();
    void createDescriptorSets();
    void updateDescriptorSets(const attachments* attachment);
};

#endif // LINK_H
