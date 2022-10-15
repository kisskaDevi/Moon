#ifndef CUSTOMFILTER_H
#define CUSTOMFILTER_H

#include "attachments.h"
#include <string>

struct CustomFilterPushConst{
    alignas (4) float deltax;
    alignas (4) float deltay;
};

class customFilter
{
private:
    VkPhysicalDevice*                   physicalDevice;
    VkDevice*                           device;
    VkQueue*                            graphicsQueue;
    VkCommandPool*                      commandPool;

    std::vector<attachments *>          Attachments;
    attachments*                        blitAttachments = nullptr;

    float                               xSampleStep = 1.5f;
    float                               ySampleStep = 1.5f;

    imageInfo                           image;

    VkRenderPass                                renderPass;
    std::vector<std::vector<VkFramebuffer>>     framebuffers;

    struct Filter{
        std::string                     ExternalPath;

        VkPipelineLayout                PipelineLayout;
        VkPipeline                      Pipeline;
        VkDescriptorSetLayout           DescriptorSetLayout;
        VkDescriptorPool                DescriptorPool;
        std::vector<VkDescriptorSet>    DescriptorSets;
        void Destroy(VkDevice* device);
        void createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void createDescriptorSetLayout(VkDevice* device);
    }filter;

public:
    customFilter();
    void destroy();

    void setExternalPath(const std::string& path);
    void setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device, VkQueue* graphicsQueue, VkCommandPool* commandPool);
    void setImageProp(imageInfo* pInfo);
    void setAttachments(uint32_t attachmentsCount, attachments* Attachments);
    void setBlitAttachments(attachments* blitAttachments);

    void setSampleStep(float deltaX, float deltaY);

    void createRenderPass();
    void createFramebuffers();
    void createPipelines();

    void createDescriptorPool();
    void createDescriptorSets();
    void updateSecondDescriptorSets();

    void render(uint32_t frameNumber, VkCommandBuffer commandBuffer, uint32_t attachmentNumber);
};


#endif // CUSTOMFILTER_H
