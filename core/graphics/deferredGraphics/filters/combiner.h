#ifndef COMBINER_H
#define COMBINER_H

#include <libs/vulkan/vulkan.h>
#include "../attachments.h"

#include <string>

class imagesCombiner
{
private:
    VkPhysicalDevice*                   physicalDevice{nullptr};
    VkDevice*                           device{nullptr};
    VkQueue*                            graphicsQueue{nullptr};
    VkCommandPool*                      commandPool{nullptr};

    imageInfo                           image;

    uint32_t                            attachmentsCount{0};
    attachments*                        pAttachments{nullptr};

    VkRenderPass                        renderPass{VK_NULL_HANDLE};
    std::vector<VkFramebuffer>          framebuffers;

    struct Combiner{
        std::string                     ExternalPath;
        uint32_t                        combineAttachmentsCount{0};

        VkPipelineLayout                PipelineLayout{VK_NULL_HANDLE};
        VkPipeline                      Pipeline{VK_NULL_HANDLE};
        VkDescriptorSetLayout           DescriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorPool                DescriptorPool{VK_NULL_HANDLE};
        std::vector<VkDescriptorSet>    DescriptorSets;
        void Destroy(VkDevice* device);
        void createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void createDescriptorSetLayout(VkDevice* device);
    }combiner;

public:
    imagesCombiner();
    void destroy();

    void setExternalPath(const std::string& path);
    void setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device, VkQueue* graphicsQueue, VkCommandPool* commandPool);
    void setImageProp(imageInfo* pInfo);

    void setAttachments(uint32_t attachmentsCount, attachments* pAttachments);
    void createAttachments(uint32_t attachmentsCount, attachments* pAttachments);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();

    void createDescriptorPool();
    void createDescriptorSets();
    void updateDescriptorSets(attachments* combinAttachments, attachment* depthAttachments, attachment* depthStencil);

    void render(uint32_t frameNumber, VkCommandBuffer commandBuffer);

    void setCombineAttachmentsCount(uint32_t combineAttachmentsCount);
};

#endif // COMBINER_H
