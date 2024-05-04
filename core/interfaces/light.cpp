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


void Light::createBufferDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout){
    std::vector<VkDescriptorSetLayoutBinding> binding;
    binding.push_back(moon::utils::vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    binding.back().stageFlags |= VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(binding.size());
        layoutInfo.pBindings = binding.data();
    CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, descriptorSetLayout));
}

void Light::createTextureDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout){
    std::vector<VkDescriptorSetLayoutBinding> binding;
        binding.push_back(moon::utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(binding.size());
        layoutInfo.pBindings = binding.data();
    CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, descriptorSetLayout));
}

}
