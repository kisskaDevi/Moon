#ifndef SSAO_H
#define SSAO_H

#include "graphics.h"
#include "attachments.h"

class SSAOGraphics
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

    struct SSAO{
        VkPipelineLayout                PipelineLayout;
        VkPipeline                      Pipeline;
        VkDescriptorSetLayout           DescriptorSetLayout;
        VkDescriptorPool                DescriptorPool;
        std::vector<VkDescriptorSet>    DescriptorSets;
        void Destroy(VkDevice* device);
        void createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void createDescriptorSetLayout(VkDevice* device);
    }ssao;

public:
    SSAOGraphics();
    void destroy();

    void setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device, VkQueue* graphicsQueue, VkCommandPool* commandPool);
    void setImageProp(imageInfo* pInfo);
    void setSSAOAttachments(attachments* Attachments);

    void createRenderPass();
    void createFramebuffers();
    void createPipelines();

    void createDescriptorPool();
    void createDescriptorSets();
    void updateSecondDescriptorSets(DeferredAttachments Attachments, VkBuffer* pUniformBuffers);

    void render(uint32_t frameNumber, VkCommandBuffer commandBuffer);
};
#endif // SSAO_H
