#ifndef OBJECT_H
#define OBJECT_H

#include <vulkan.h>
#include <glm.hpp>
#include <vector>

class model;
class cubeTexture;

class object
{
public:
    virtual ~object(){};
    virtual void destroy(VkDevice device) = 0;

    virtual uint8_t getPipelineBitMask() const = 0;

    virtual std::vector<VkDescriptorSet>&   getDescriptorSet() = 0;

    virtual void createUniformBuffers(VkPhysicalDevice physicalDevice, VkDevice device, uint32_t imageCount) = 0;
    virtual void updateUniformBuffer(VkCommandBuffer commandBuffer, uint32_t frameNumber) = 0;

    virtual model* getModel() = 0;
    virtual uint32_t getInstanceNumber(uint32_t imageNumber) const = 0;

    virtual bool getEnable() const = 0;
    virtual bool getEnableShadow() const = 0;

    virtual bool getOutliningEnable() const = 0;
    virtual float getOutliningWidth() const = 0;
    virtual glm::vec4 getOutliningColor() const = 0;

    virtual void setFirstPrimitive(uint32_t firstPrimitive) = 0;
    virtual void setPrimitiveCount(uint32_t primitiveCount) = 0;
    virtual void resetPrimitiveCount() = 0;

    virtual uint32_t getFirstPrimitive() const = 0;

    virtual void createDescriptorPool(VkDevice device, uint32_t imageCount) = 0;
    virtual void createDescriptorSet(VkDevice device, uint32_t imageCount) = 0;

    virtual cubeTexture* getTexture() = 0;

    static void createDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout);
    static void createSkyboxDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout);
};

#endif // OBJECT_H
