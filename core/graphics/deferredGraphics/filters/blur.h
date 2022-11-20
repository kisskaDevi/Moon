#ifndef BLUR_H
#define BLUR_H

#include <libs/vulkan/vulkan.h>
#include "../attachments.h"

#include <string>

class gaussianBlur
{
private:
    VkPhysicalDevice*                   physicalDevice{nullptr};
    VkDevice*                           device{nullptr};
    VkQueue*                            graphicsQueue{nullptr};
    VkCommandPool*                      commandPool{nullptr};

    imageInfo                           image;

    uint32_t                            attachmentsCount{0};
    attachments*                        pAttachments{nullptr};

    attachments                         bufferAttachment;

    VkRenderPass                        renderPass{VK_NULL_HANDLE};
    std::vector<VkFramebuffer>          framebuffers;

    struct xBlur{
        std::string                     ExternalPath;

        VkPipelineLayout                PipelineLayout{VK_NULL_HANDLE};
        VkPipeline                      Pipeline{VK_NULL_HANDLE};
        VkDescriptorSetLayout           DescriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorPool                DescriptorPool{VK_NULL_HANDLE};
        std::vector<VkDescriptorSet>    DescriptorSets{VK_NULL_HANDLE};
        void Destroy(VkDevice* device);
        void createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void createDescriptorSetLayout(VkDevice* device);
    }xblur;

    struct yBlur{
        std::string                     ExternalPath;

        VkPipelineLayout                PipelineLayout{VK_NULL_HANDLE};
        VkPipeline                      Pipeline{VK_NULL_HANDLE};
        VkDescriptorSetLayout           DescriptorSetLayout{VK_NULL_HANDLE};
        VkDescriptorPool                DescriptorPool{VK_NULL_HANDLE};
        std::vector<VkDescriptorSet>    DescriptorSets;
        void Destroy(VkDevice* device);
        void createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void createDescriptorSetLayout(VkDevice* device);
    }yblur;

public:
    gaussianBlur();
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
    void updateDescriptorSets(attachments* blurAttachment);

    void render(uint32_t frameNumber, VkCommandBuffer commandBuffer);

    void createBufferAttachments();
};

#endif // BLUR_H
