#include "object.h"

void object::createDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout){
    std::vector<VkDescriptorSetLayoutBinding> binding;
    binding.push_back(VkDescriptorSetLayoutBinding{});
        binding.back().binding = binding.size() - 1;
        binding.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        binding.back().descriptorCount = 1;
        binding.back().stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        binding.back().pImmutableSamplers = nullptr;
    VkDescriptorSetLayoutCreateInfo uniformBufferLayoutInfo{};
        uniformBufferLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        uniformBufferLayoutInfo.bindingCount = static_cast<uint32_t>(binding.size());
        uniformBufferLayoutInfo.pBindings = binding.data();
    vkCreateDescriptorSetLayout(device, &uniformBufferLayoutInfo, nullptr, descriptorSetLayout);
}

void object::createSkyboxDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout)
{
    std::vector<VkDescriptorSetLayoutBinding> binding;
    binding.push_back(VkDescriptorSetLayoutBinding{});
        binding.back().binding = binding.size() - 1;
        binding.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        binding.back().descriptorCount = 1;
        binding.back().stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        binding.back().pImmutableSamplers = nullptr;
    binding.push_back(VkDescriptorSetLayoutBinding{});
        binding.back().binding = binding.size() - 1;
        binding.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binding.back().descriptorCount = 1;
        binding.back().stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        binding.back().pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo uniformBufferLayoutInfo{};
        uniformBufferLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        uniformBufferLayoutInfo.bindingCount = static_cast<uint32_t>(binding.size());
        uniformBufferLayoutInfo.pBindings = binding.data();
    vkCreateDescriptorSetLayout(device, &uniformBufferLayoutInfo, nullptr, descriptorSetLayout);
}
