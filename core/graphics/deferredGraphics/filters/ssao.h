#ifndef SSAO_H
#define SSAO_H

#include <libs/vulkan/vulkan.h>
#include "../attachments.h"

#include <string>

class SSAOGraphics
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

    struct SSAO{
        std::string                     ExternalPath;

        VkPipelineLayout                PipelineLayout;
        VkPipeline                      Pipeline;
        VkDescriptorSetLayout           DescriptorSetLayout;
        VkDescriptorPool                DescriptorPool;
        std::vector<VkDescriptorSet>    DescriptorSets;
        void Destroy(VkDevice* device);
        void createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass);
        void createDescriptorSetLayout(VkDevice* device);
    }ssao;

    void setAttachments(uint32_t attachmentsCount, attachments* Attachments);
public:
    SSAOGraphics();
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
    void updateDescriptorSets(DeferredAttachments Attachments, VkBuffer* pUniformBuffers);

    void render(uint32_t frameNumber, VkCommandBuffer commandBuffer);
};
#endif // SSAO_H
