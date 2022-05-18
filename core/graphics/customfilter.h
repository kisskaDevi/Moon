#ifndef CUSTOMFILTER_H
#define CUSTOMFILTER_H

#include "attachments.h"

class                               VkApplication;

struct CustomFilterPushConst{
    alignas (4) float deltax;
    alignas (4) float deltay;
};

class customFilter
{
private:
    VkApplication                       *app;
    std::vector<attachments>            *Attachments;
    attachments                         *blitAttachments;

    struct Image{
        uint32_t                        Count;
        VkFormat                        Format;
        VkExtent2D                      Extent;
        VkSampleCountFlagBits           Samples = VK_SAMPLE_COUNT_1_BIT;
    }image;

    VkRenderPass                                renderPass;
    std::vector<std::vector<VkFramebuffer>>     framebuffers;

    struct Filter{
        VkPipelineLayout                PipelineLayout;
        VkPipeline                      Pipeline;
        VkDescriptorSetLayout           DescriptorSetLayout;
        VkDescriptorPool                DescriptorPool;
        std::vector<VkDescriptorSet>    DescriptorSets;
        void Destroy(VkApplication* app);
        void createPipeline(VkApplication* app, Image* image, VkRenderPass* renderPass);
        void createDescriptorSetLayout(VkApplication* app);
    }filter;

public:
    customFilter();
    void destroy();

    void setApplication(VkApplication *app);
    void setImageProp(uint32_t imageCount, VkFormat imageFormat, VkExtent2D imageExtent, VkSampleCountFlagBits imageSamples);
    void setAttachments(std::vector<attachments>* Attachments);
    void setBlitAttachments(attachments* blitAttachments);

    void createRenderPass();
    void createFramebuffers();
    void createPipelines();

    void createDescriptorPool();
    void createDescriptorSets();
    void updateSecondDescriptorSets();

    void render(std::vector<VkCommandBuffer> &commandBuffers, uint32_t i, uint32_t j, float delta);

};

#endif // CUSTOMFILTER_H
