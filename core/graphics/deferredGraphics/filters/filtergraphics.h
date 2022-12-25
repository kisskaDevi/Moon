#ifndef FILTERGRAPHICS_H
#define FILTERGRAPHICS_H

#include <libs/vulkan/vulkan.h>
#include "../attachments.h"

#include <string>

class filter{
    virtual void Destroy(VkDevice* device) = 0;
    virtual void createPipeline(VkDevice* device, imageInfo* pInfo, VkRenderPass* pRenderPass) = 0;
    virtual void createDescriptorSetLayout(VkDevice* device) = 0;
};

class filterGraphics
{
public:
    virtual ~filterGraphics(){};
    virtual void destroy() = 0;

    virtual void setExternalPath(const std::string& path) = 0;
    virtual void setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device, VkQueue* graphicsQueue, VkCommandPool* commandPool) = 0;
    virtual void setImageProp(imageInfo* pInfo) = 0;

    virtual void setAttachments(uint32_t attachmentsCount, attachments* pAttachments) = 0;
    virtual void createAttachments(uint32_t attachmentsCount, attachments* pAttachments) = 0;
    virtual void createRenderPass() = 0;
    virtual void createFramebuffers() = 0;
    virtual void createPipelines() = 0;

    virtual void createDescriptorPool() = 0;
    virtual void createDescriptorSets() = 0;

    virtual void render(uint32_t frameNumber, VkCommandBuffer commandBuffer) = 0;
};

#endif // FILTERGRAPHICS_H
