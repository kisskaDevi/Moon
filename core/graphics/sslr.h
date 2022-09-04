#ifndef SSLR_H
#define SSLR_H

#include "graphics.h"

class SSLRGraphics
{
private:
    VkPhysicalDevice*                   physicalDevice;
    VkDevice*                           device;
    VkQueue*                            graphicsQueue;
    VkCommandPool*                      commandPool;

    imageInfo                           image;

    attachments*                        Attachments = nullptr;

    VkRenderPass                        renderPass;
    std::vector<VkFramebuffer>          framebuffers;

    struct SSLR{
        VkPipelineLayout                PipelineLayout;
        VkPipeline                      Pipeline;
        VkDescriptorSetLayout           DescriptorSetLayout;
        VkDescriptorPool                DescriptorPool;
        std::vector<VkDescriptorSet>    DescriptorSets;
        void Destroy(VkDevice* device);
        void createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void createDescriptorSetLayout(VkDevice* device);
    }sslr;

public:
    SSLRGraphics();
    void destroy();

    void setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device, VkQueue* graphicsQueue, VkCommandPool* commandPool);
    void setImageProp(imageInfo* pInfo);
    void setSSLRAttachments(attachments* Attachments);

    void createRenderPass();
    void createFramebuffers();
    void createPipelines();

    void createDescriptorPool();
    void createDescriptorSets();
    void updateSecondDescriptorSets(DeferredAttachments Attachments, VkBuffer* pUniformBuffers);

    void render(uint32_t frameNumber, VkCommandBuffer commandBuffer);
};

#endif // SSLR_H
