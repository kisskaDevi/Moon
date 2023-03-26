#include "model.h"
#include "../utils/vkdefault.h"

VkVertexInputBindingDescription model::Vertex::getBindingDescription(){
    return VkVertexInputBindingDescription{0,sizeof(Vertex),VK_VERTEX_INPUT_RATE_VERTEX};
}

std::vector<VkVertexInputAttributeDescription> model::Vertex::getAttributeDescriptions(){
    std::vector<VkVertexInputAttributeDescription> attributeDescriptions;
    attributeDescriptions.push_back(VkVertexInputAttributeDescription{static_cast<uint32_t>(attributeDescriptions.size()),0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex, pos)});
    attributeDescriptions.push_back(VkVertexInputAttributeDescription{static_cast<uint32_t>(attributeDescriptions.size()),0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex, normal)});
    attributeDescriptions.push_back(VkVertexInputAttributeDescription{static_cast<uint32_t>(attributeDescriptions.size()),0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex, uv0)});
    attributeDescriptions.push_back(VkVertexInputAttributeDescription{static_cast<uint32_t>(attributeDescriptions.size()),0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex, uv1)});
    attributeDescriptions.push_back(VkVertexInputAttributeDescription{static_cast<uint32_t>(attributeDescriptions.size()),0,VK_FORMAT_R32G32B32A32_SFLOAT,offsetof(Vertex, joint0)});
    attributeDescriptions.push_back(VkVertexInputAttributeDescription{static_cast<uint32_t>(attributeDescriptions.size()),0,VK_FORMAT_R32G32B32A32_SFLOAT,offsetof(Vertex, weight0)});
    attributeDescriptions.push_back(VkVertexInputAttributeDescription{static_cast<uint32_t>(attributeDescriptions.size()),0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex, tangent)});
    attributeDescriptions.push_back(VkVertexInputAttributeDescription{static_cast<uint32_t>(attributeDescriptions.size()),0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex, bitangent)});

    return attributeDescriptions;
}

void model::createNodeDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout)
{
    std::vector<VkDescriptorSetLayoutBinding> binding;
    binding.push_back(vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    VkDescriptorSetLayoutCreateInfo uniformBlockLayoutInfo{};
        uniformBlockLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        uniformBlockLayoutInfo.bindingCount = static_cast<uint32_t>(binding.size());
        uniformBlockLayoutInfo.pBindings = binding.data();
    vkCreateDescriptorSetLayout(device, &uniformBlockLayoutInfo, nullptr, descriptorSetLayout);
}

void model::createMaterialDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout)
{
    std::vector<VkDescriptorSetLayoutBinding> binding;
    binding.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    binding.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    binding.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    binding.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    binding.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    VkDescriptorSetLayoutCreateInfo materialLayoutInfo{};
        materialLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        materialLayoutInfo.bindingCount = static_cast<uint32_t>(binding.size());
        materialLayoutInfo.pBindings = binding.data();
    vkCreateDescriptorSetLayout(device, &materialLayoutInfo, nullptr, descriptorSetLayout);
}
