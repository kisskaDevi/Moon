#ifndef LIGHT_H
#define LIGHT_H

#include <vector>
#include <unordered_map>

#include <vulkan.h>
#include <vkdefault.h>
#include <device.h>
#include <depthMap.h>

namespace moon::interfaces {

enum LightType : uint8_t {
    spot = 0x1
};

class Light
{
protected:
    bool enableShadow{false};
    bool enableScattering{false};

    moon::utils::vkDefault::DescriptorSetLayout descriptorSetLayout;
    moon::utils::vkDefault::DescriptorPool descriptorPool;
    moon::utils::vkDefault::DescriptorSets descriptorSets;

    uint8_t pipelineBitMask{0};

public:
    virtual ~Light(){};

    void setEnableShadow(bool enable);
    void setEnableScattering(bool enable);

    bool isShadowEnable() const;
    bool isScatteringEnable() const;

    uint8_t getPipelineBitMask() const;
    const moon::utils::vkDefault::DescriptorSets& getDescriptorSets() const;

    virtual void create(
            const moon::utils::PhysicalDevice& device,
            VkCommandPool commandPool,
            uint32_t imageCount) = 0;

    virtual void render(
        uint32_t frameNumber,
        VkCommandBuffer commandBuffer,
        const utils::vkDefault::DescriptorSets& descriptorSet,
        VkPipelineLayout pipelineLayout,
        VkPipeline pipeline) = 0;

    virtual void update(
        uint32_t frameNumber,
        VkCommandBuffer commandBuffer) = 0;

    static moon::utils::vkDefault::DescriptorSetLayout createTextureDescriptorSetLayout(VkDevice device);
    static moon::utils::vkDefault::DescriptorSetLayout createBufferDescriptorSetLayout(VkDevice device);
};

using Lights = std::vector<moon::interfaces::Light*>;
using DepthMaps = std::unordered_map<moon::interfaces::Light*, moon::utils::DepthMap>;

}
#endif // LIGHT_H
