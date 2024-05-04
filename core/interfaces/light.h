#ifndef LIGHT_H
#define LIGHT_H

#include <vulkan.h>
#include <vector>

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

    VkDescriptorSetLayout               descriptorSetLayout{VK_NULL_HANDLE};
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
            moon::utils::PhysicalDevice device,
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

    static void createTextureDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout);
    static void createBufferDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout);
};

}
#endif // LIGHT_H
