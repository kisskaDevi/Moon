#ifndef COMBINER_H
#define COMBINER_H

#include <libs/vulkan/vulkan.h>
#include "../attachments.h"

#include <string>

class imagesCombiner
{
private:
    VkPhysicalDevice*                   physicalDevice;
    VkDevice*                           device;
    VkQueue*                            graphicsQueue;
    VkCommandPool*                      commandPool;

    imageInfo                           image;

    uint32_t                            attachmentsCount = 0;
    attachments*                        Attachments = nullptr;

    VkRenderPass                        renderPass;
    std::vector<VkFramebuffer>          framebuffers;

    struct Combiner{
        std::string                     ExternalPath;
        uint32_t                        combineAttachmentsCount = 0;

        VkPipelineLayout                PipelineLayout;
        VkPipeline                      Pipeline;
        VkDescriptorSetLayout           DescriptorSetLayout;
        VkDescriptorPool                DescriptorPool;
        std::vector<VkDescriptorSet>    DescriptorSets;
        void Destroy(VkDevice* device);
        void createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void createDescriptorSetLayout(VkDevice* device);
    }combiner;

    void setAttachments(uint32_t attachmentsCount, attachments* Attachments);
public:
    imagesCombiner();
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
    void updateDescriptorSets(attachments* Attachments, attachment* depthAttachments, attachment* depthStencil);

    void render(uint32_t frameNumber, VkCommandBuffer commandBuffer);

    void setCombineAttachmentsCount(uint32_t combineAttachmentsCount);
};

#endif // COMBINER_H
