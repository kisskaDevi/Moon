#ifndef LIGHTINTERFACE_H
#define LIGHTINTERFACE_H

#include "libs/vulkan/vulkan.h"
#include "core/graphics/deferredGraphics/attachments.h"

#include <vector>
#include <string>

class texture;
class attachments;

class light
{
public:
    virtual ~light(){};

    virtual void destroy(VkDevice* device) = 0;

    virtual texture* getTexture() = 0;
    virtual attachments* getAttachments() = 0;
    virtual uint8_t getPipelineBitMask() = 0;

    virtual bool isShadowEnable() const = 0;

    virtual VkDescriptorSet* getDescriptorSets() = 0;
    virtual VkDescriptorSet* getShadowDescriptorSets() = 0;

    virtual void createUniformBuffers(VkPhysicalDevice* physicalDevice, VkDevice* device, uint32_t imageCount) = 0;
    virtual void updateUniformBuffer(VkCommandBuffer commandBuffer, uint32_t frameNumber) = 0;

    virtual void createDescriptorPool(VkDevice* device, uint32_t imageCount) = 0;
    virtual void createDescriptorSets(VkDevice* device, uint32_t imageCount) = 0;
    virtual void updateDescriptorSets(VkDevice* device, uint32_t imageCount, texture* emptyTexture) = 0;
};

namespace SpotLight {
    void createDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout);
    void createShadowDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout);
}

#endif // LIGHTINTERFACE_H
