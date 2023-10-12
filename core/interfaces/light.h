#ifndef LIGHT_H
#define LIGHT_H

#include <vulkan.h>

class texture;
class attachments;
struct physicalDevice;

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

    virtual void create(
            physicalDevice device,
            VkCommandPool commandPool,
            uint32_t imageCount,
            texture* emptyTextureBlack = nullptr,
            texture* emptyTextureWhite = nullptr) = 0;

    virtual void updateUniformBuffer(VkCommandBuffer commandBuffer, uint32_t frameNumber) = 0;

    static void createTextureDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout);
    static void createBufferDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout);
};

#endif // LIGHT_H
