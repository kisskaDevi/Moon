#ifndef LIGHT_H
#define LIGHT_H

#include <vector>
#include <unordered_map>

#include <vulkan.h>
#include <vkdefault.h>
#include <device.h>
#include <depthMap.h>

namespace moon::interfaces {

class Light
{
protected:
    uint8_t pipelineBitMask{ 0 };
    bool enableShadow{false};
    bool enableScattering{false};

    moon::utils::vkDefault::DescriptorSetLayout descriptorSetLayout;
    moon::utils::vkDefault::DescriptorPool descriptorPool;
    moon::utils::vkDefault::DescriptorSets descriptorSets;

public:
    enum Type : uint8_t {
        spot = 0x1
    };

    Light(uint8_t pipelineBitMask, bool enableShadow, bool enableScattering);

    virtual ~Light(){};

    void setEnableShadow(bool enable);
    void setEnableScattering(bool enable);

    bool isShadowEnable() const;
    bool isScatteringEnable() const;

    uint8_t& pipelineFlagBits();
    const VkDescriptorSet& getDescriptorSet(uint32_t i) const;

    virtual utils::Buffers& buffers() = 0;
    virtual void create(const moon::utils::PhysicalDevice& device, VkCommandPool commandPool, uint32_t imageCount) = 0;
    virtual void update(uint32_t frameNumber, VkCommandBuffer commandBuffer) = 0;
    virtual void render(uint32_t frameNumber, VkCommandBuffer commandBuffer, const utils::vkDefault::DescriptorSets& descriptorSet, VkPipelineLayout pipelineLayout, VkPipeline pipeline) = 0;

    static moon::utils::vkDefault::DescriptorSetLayout createTextureDescriptorSetLayout(VkDevice device);
    static moon::utils::vkDefault::DescriptorSetLayout createBufferDescriptorSetLayout(VkDevice device);
};

using Lights = std::vector<moon::interfaces::Light*>;
using DepthMaps = std::unordered_map<moon::interfaces::Light*, moon::utils::DepthMap>;

}
#endif // LIGHT_H
