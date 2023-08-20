#ifndef LIGHT_H
#define LIGHT_H

#include <vulkan.h>

class texture;
class attachments;

class light
{
public:
    virtual ~light(){};

    virtual void destroy(VkDevice device) = 0;

    virtual texture* getTexture() = 0;
    virtual attachments* getAttachments() = 0;
    virtual uint8_t getPipelineBitMask() const = 0;

    virtual bool isShadowEnable() const = 0;
    virtual bool isScatteringEnable() const = 0;

    virtual VkDescriptorSet* getDescriptorSets() = 0;
    virtual VkDescriptorSet* getBufferDescriptorSets() = 0;

    virtual void createUniformBuffers(VkPhysicalDevice physicalDevice, VkDevice device, uint32_t imageCount) = 0;
    virtual void updateUniformBuffer(VkCommandBuffer commandBuffer, uint32_t frameNumber) = 0;

    virtual void createDescriptorPool(VkDevice device, uint32_t imageCount) = 0;
    virtual void createDescriptorSets(VkDevice device, uint32_t imageCount) = 0;
    virtual void updateDescriptorSets(VkDevice device, uint32_t imageCount, texture* emptyTextureBlack , texture* emptyTextureWhite) = 0;

    static void createTextureDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout);
    static void createBufferDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout);
};

#endif // LIGHT_H
