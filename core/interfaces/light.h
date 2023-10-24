#ifndef LIGHT_H
#define LIGHT_H

#include <vulkan.h>
#include <unordered_map>

class texture;
class attachments;
struct physicalDevice;

class light
{
protected:
    bool enableShadow{false};
    bool enableScattering{false};

public:
    virtual ~light(){};

    void setEnableShadow(bool enable);
    void setEnableScattering(bool enable);

    bool isShadowEnable() const;
    bool isScatteringEnable() const;

    virtual void destroy(VkDevice device) = 0;

    virtual attachments* getAttachments() = 0;
    virtual const std::vector<VkDescriptorSet>& getDescriptorSets() const = 0;
    virtual uint8_t getPipelineBitMask() const = 0;

    virtual void create(
            physicalDevice device,
            VkCommandPool commandPool,
            uint32_t imageCount) = 0;

    virtual void render(
        uint32_t frameNumber,
        VkCommandBuffer commandBuffer,
        VkDescriptorSet descriptorSet,
        std::unordered_map<uint8_t, VkPipelineLayout> PipelineLayoutDictionary,
        std::unordered_map<uint8_t, VkPipeline> PipelinesDictionary) = 0;

    virtual void update(
        uint32_t frameNumber,
        VkCommandBuffer commandBuffer) = 0;

    static void createTextureDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout);
    static void createBufferDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout);
};

#endif // LIGHT_H
