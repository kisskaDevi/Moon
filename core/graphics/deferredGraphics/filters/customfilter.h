#ifndef CUSTOMFILTER_H
#define CUSTOMFILTER_H

#include <libs/vulkan/vulkan.h>
#include "../attachments.h"

#include <string>
#include <vector>

class customFilter
{
private:
    VkPhysicalDevice*                   physicalDevice;
    VkDevice*                           device;
    VkQueue*                            graphicsQueue;
    VkCommandPool*                      commandPool;

    imageInfo                           image;

    uint32_t                            attachmentsCount = 0;
    attachments*                        Attachments = nullptr;

    attachments                         bufferAttachment;
    attachments*                        srcAttachment;
    float                               blitFactor;
    float                               xSampleStep = 1.5f;
    float                               ySampleStep = 1.5f;

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

    void setAttachments(uint32_t attachmentsCount, attachments* Attachments);
    void render(uint32_t frameNumber, VkCommandBuffer commandBuffer, uint32_t attachmentNumber);
public:
    customFilter();
    void destroy();

    void setExternalPath(const std::string& path);
    void setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device, VkQueue* graphicsQueue, VkCommandPool* commandPool);
    void setImageProp(imageInfo* pInfo);

    void createAttachments(uint32_t attachmentsCount, attachments* Attachments);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();

    void createDescriptorPool();
    void createDescriptorSets();
    void updateDescriptorSets();

    void render(uint32_t frameNumber, VkCommandBuffer commandBuffer);

    void createBufferAttachments();
    void setSrcAttachment(attachments* srcAttachment);
    void setBlitFactor(const float& blitFactor);
    void setSampleStep(float deltaX, float deltaY);
};


#endif // CUSTOMFILTER_H
