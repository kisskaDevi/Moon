#include "light.h"
#include "vkdefault.h"
#include "operations.h"
#include <vector>

namespace moon::interfaces {

void Light::setEnableShadow(bool enable){
    enableShadow = enable;
}

void Light::setEnableScattering(bool enable){
    enableScattering = enable;
}

bool Light::isShadowEnable() const{
    return enableShadow;
}

bool Light::isScatteringEnable() const{
    return enableScattering;
}

const std::vector<VkDescriptorSet>& Light::getDescriptorSets() const {
    return descriptorSets;
}

uint8_t Light::getPipelineBitMask() const {
    return pipelineBitMask;
}

moon::utils::vkDefault::DescriptorSetLayout Light::createBufferDescriptorSetLayout(VkDevice device){
    moon::utils::vkDefault::DescriptorSetLayout descriptorSetLayout;

    std::vector<VkDescriptorSetLayoutBinding> binding;
        binding.push_back(moon::utils::vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
        binding.back().stageFlags |= VK_SHADER_STAGE_FRAGMENT_BIT;

    CHECK(descriptorSetLayout.create(device, binding));
    return descriptorSetLayout;
}

moon::utils::vkDefault::DescriptorSetLayout Light::createTextureDescriptorSetLayout(VkDevice device){
    moon::utils::vkDefault::DescriptorSetLayout descriptorSetLayout;

    std::vector<VkDescriptorSetLayoutBinding> binding;
        binding.push_back(moon::utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));

    CHECK(descriptorSetLayout.create(device, binding));
    return descriptorSetLayout;
}

}
