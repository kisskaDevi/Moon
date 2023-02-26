#ifndef FILTERGRAPHICS_H
#define FILTERGRAPHICS_H

#include <libs/vulkan/vulkan.h>
#include "../attachments.h"

#include <string>

class texture;

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

    virtual void setEmptyTexture(texture* emptyTexture) = 0;
    virtual void setExternalPath(const std::string& path) = 0;
    virtual void setDeviceProp(VkPhysicalDevice* physicalDevice, VkDevice* device) = 0;
    virtual void setImageProp(imageInfo* pInfo) = 0;

    virtual void setAttachments(uint32_t attachmentsCount, attachments* pAttachments) = 0;
    virtual void createAttachments(uint32_t attachmentsCount, attachments* pAttachments) = 0;
    virtual void createRenderPass() = 0;
    virtual void createFramebuffers() = 0;
    virtual void createPipelines() = 0;

    virtual void createDescriptorPool() = 0;
    virtual void createDescriptorSets() = 0;

    virtual void createCommandBuffers(VkCommandPool commandPool) = 0;
    virtual void updateCommandBuffer(uint32_t frameNumber) = 0;
    virtual VkCommandBuffer& getCommandBuffer(uint32_t frameNumber) = 0;
};

#endif // FILTERGRAPHICS_H
