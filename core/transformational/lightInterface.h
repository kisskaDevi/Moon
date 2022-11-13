#ifndef LIGHTINTERFACE_H
#define LIGHTINTERFACE_H

#include "libs/vulkan/vulkan.h"

#include <vector>
#include <string>

class object;
class texture;
class QueueFamilyIndices;

class light
{
public:
    virtual ~light(){};

    virtual void destroyUniformBuffers(VkDevice* device) = 0;
    virtual void destroy(VkDevice* device) = 0;

    virtual texture* getTexture() = 0;

    virtual uint8_t getPipelineBitMask() = 0;

    virtual bool isShadowEnable() const = 0;

    virtual VkDescriptorSet* getDescriptorSets() = 0;
    virtual VkCommandBuffer* getShadowCommandBuffer() = 0;

    virtual void createUniformBuffers(VkPhysicalDevice* physicalDevice, VkDevice* device, uint32_t imageCount) = 0;
    virtual void updateUniformBuffer(VkDevice* device, uint32_t frameNumber) = 0;

    virtual void createShadow(VkPhysicalDevice* physicalDevice, VkDevice* device, QueueFamilyIndices* queueFamilyIndices, uint32_t imageCount, const std::string& ExternalPath) = 0;
    virtual void updateShadowDescriptorSets() = 0;
    virtual void createShadowCommandBuffers() = 0;
    virtual void updateShadowCommandBuffer(uint32_t frameNumber, std::vector<object*>& objects) = 0;

    virtual void createDescriptorPool(VkDevice* device, uint32_t imageCount) = 0;
    virtual void createDescriptorSets(VkDevice* device, uint32_t imageCount) = 0;
    virtual void updateDescriptorSets(VkDevice* device, uint32_t imageCount, texture* emptyTexture) = 0;
};

#endif // LIGHTINTERFACE_H
