#include "light.h"
#include "vkdefault.h"
#include "operations.h"
#include <vector>

void light::setEnableShadow(bool enable){
    enableShadow = enable;
}

void light::setEnableScattering(bool enable){
    enableScattering = enable;
}

bool light::isShadowEnable() const{
    return enableShadow;
}

bool light::isScatteringEnable() const{
    return enableScattering;
}

void light::createBufferDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout){
    std::vector<VkDescriptorSetLayoutBinding> binding;
    binding.push_back(vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    binding.back().stageFlags |= VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(binding.size());
        layoutInfo.pBindings = binding.data();
    CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, descriptorSetLayout));
}

void light::createTextureDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout){
    std::vector<VkDescriptorSetLayoutBinding> binding;
        binding.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(binding.size());
        layoutInfo.pBindings = binding.data();
    CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, descriptorSetLayout));
}

