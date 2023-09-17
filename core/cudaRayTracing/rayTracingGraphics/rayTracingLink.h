#ifndef RAYTRACINGLINK
#define RAYTRACINGLINK

#include <vector>
#include <filesystem>

#include "linkable.h"
#include "core/utils/attachments.h"

class rayTracingLink : public linkable{
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

public:
    rayTracingLink() = default;
    void destroy();
    void setDeviceProp(VkDevice device);
    void setImageCount(const uint32_t& count);
    void setShadersPath(const std::filesystem::path& shadersPath);

    void draw(VkCommandBuffer commandBuffer, uint32_t imageNumber) const override;
    void setRenderPass(VkRenderPass renderPass) override;

    void createDescriptorSetLayout();
    void createPipeline(imageInfo* pInfo);
    void createDescriptorPool();
    void createDescriptorSets();
    void updateDescriptorSets(attachments* attachment);;
};

#endif
