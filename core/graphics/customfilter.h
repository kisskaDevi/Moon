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
    VkApplication*                      app;
    std::vector<attachments *>          Attachments;
    attachments*                        blitAttachments;

    imageInfo                           image;

    VkRenderPass                                renderPass;
    std::vector<std::vector<VkFramebuffer>>     framebuffers;

    struct Filter{
        VkPipelineLayout                PipelineLayout;
        VkPipeline                      Pipeline;
        VkDescriptorSetLayout           DescriptorSetLayout;
        VkDescriptorPool                DescriptorPool;
        std::vector<VkDescriptorSet>    DescriptorSets;
        void Destroy(VkApplication* app);
        void createPipeline(VkApplication* app, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void createDescriptorSetLayout(VkApplication* app);
    }filter;

public:
    customFilter();
    void destroy();

    void setApplication(VkApplication *app);
    void setImageProp(imageInfo* pInfo);
    void setAttachments(uint32_t attachmentsCount, attachments* Attachments);
    void setBlitAttachments(attachments* blitAttachments);

    void createRenderPass();
    void createFramebuffers();
    void createPipelines();

    void createDescriptorPool();
    void createDescriptorSets();
    void updateSecondDescriptorSets();

    void render(uint32_t frameNumber, VkCommandBuffer commandBuffer, uint32_t attachmentNumber, float delta);

};

#endif // CUSTOMFILTER_H
