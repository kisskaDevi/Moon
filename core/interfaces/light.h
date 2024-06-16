#ifndef LIGHT_H
#define LIGHT_H

#include <vulkan.h>
#include <vector>
#include <vkdefault.h>

namespace moon::utils { struct PhysicalDevice;}

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
    VkDescriptorPool                    descriptorPool{VK_NULL_HANDLE};
    std::vector<VkDescriptorSet>        descriptorSets;

    uint8_t pipelineBitMask{0};

public:
    virtual ~Light(){};

    void setEnableShadow(bool enable);
    void setEnableScattering(bool enable);

    bool isShadowEnable() const;
    bool isScatteringEnable() const;

    uint8_t getPipelineBitMask() const;
    const std::vector<VkDescriptorSet>& getDescriptorSets() const;

    virtual void destroy(VkDevice device) = 0;

    virtual void create(
            const moon::utils::PhysicalDevice& device,
            VkCommandPool commandPool,
            uint32_t imageCount) = 0;

    virtual void render(
        uint32_t frameNumber,
        VkCommandBuffer commandBuffer,
        const std::vector<VkDescriptorSet>& descriptorSet,
        VkPipelineLayout pipelineLayout,
        VkPipeline pipeline) = 0;

    virtual void update(
        uint32_t frameNumber,
        VkCommandBuffer commandBuffer) = 0;

    static moon::utils::vkDefault::DescriptorSetLayout createTextureDescriptorSetLayout(VkDevice device);
    static moon::utils::vkDefault::DescriptorSetLayout createBufferDescriptorSetLayout(VkDevice device);
};

}
#endif // LIGHT_H
