#include "light.h"

#include <vector>

void light::createBufferDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout){
    std::vector<VkDescriptorSetLayoutBinding> binding;
    binding.push_back(VkDescriptorSetLayoutBinding{});
        binding.back().binding = binding.size() - 1;
        binding.back().descriptorCount = 1;
        binding.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        binding.back().stageFlags = VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT;
        binding.back().pImmutableSamplers = nullptr;
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(binding.size());
        layoutInfo.pBindings = binding.data();
    vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, descriptorSetLayout);
}

void light::createTextureDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout){
    std::vector<VkDescriptorSetLayoutBinding> binding;
    binding.push_back(VkDescriptorSetLayoutBinding{});
        binding.back().binding = binding.size() - 1;
        binding.back().descriptorCount = 1;
        binding.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binding.back().stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        binding.back().pImmutableSamplers = nullptr;
    binding.push_back(VkDescriptorSetLayoutBinding{});
        binding.back().binding = binding.size() - 1;
        binding.back().descriptorCount = 1;
        binding.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binding.back().stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        binding.back().pImmutableSamplers = nullptr;
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(binding.size());
        layoutInfo.pBindings = binding.data();
    vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, descriptorSetLayout);
}
