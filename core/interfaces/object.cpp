#include "object.h"
#include "vkdefault.h"
#include "operations.h"

bool object::getEnable() const {
    return enable;
}

bool object::getEnableShadow() const {
    return enableShadow;
}

void object::setEnable(const bool& enable) {
    this->enable = enable;
}

void object::setEnableShadow(const bool& enable) {
    this->enableShadow = enable;
}

void object::setModel(model* model, uint32_t firstInstance, uint32_t instanceCount){
    this->pModel = model;
    this->firstInstance = firstInstance;
    this->instanceCount = instanceCount;
}

model* object::getModel() {
    return pModel;
}

uint32_t object::getInstanceNumber(uint32_t imageNumber) const {
    return firstInstance + (instanceCount > imageNumber ? imageNumber : 0);
}

void object::setOutlining(const bool& enable, const float& width, const vector<float,4>& color){
    outlining.Enable = enable;
    outlining.Width = width > 0.0f ? width : outlining.Width;
    outlining.Color = dot(color,color) > 0.0f ? color : outlining.Color;

    pipelineBitMask |= outlining.Enable ? objectProperty::outlining : objectProperty::non;
}

bool object::getOutliningEnable() const {
    return outlining.Enable;
}

float object::getOutliningWidth() const {
    return outlining.Width;
}

vector<float,4> object::getOutliningColor() const {
    return outlining.Color;
}

void object::setFirstPrimitive(uint32_t firstPrimitive) {
    this->firstPrimitive = firstPrimitive;
}

void object::setPrimitiveCount(uint32_t primitiveCount) {
    this->primitiveCount = primitiveCount;
}

bool object::comparePrimitive(uint32_t primitive) {
    return !(primitive < firstPrimitive) && (primitive < firstPrimitive + primitiveCount);
}

uint32_t object::getFirstPrimitive() const {
    return firstPrimitive;
}

uint32_t object::getPrimitiveCount() const {
    return primitiveCount;
}

const std::vector<VkDescriptorSet> &object::getDescriptorSet() const {
    return descriptors;
}

uint8_t object::getPipelineBitMask() const {
    return pipelineBitMask;
}

void object::createDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout){
    std::vector<VkDescriptorSetLayoutBinding> binding;
    binding.push_back(vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    binding.back().stageFlags |= VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo uniformBufferLayoutInfo{};
        uniformBufferLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        uniformBufferLayoutInfo.bindingCount = static_cast<uint32_t>(binding.size());
        uniformBufferLayoutInfo.pBindings = binding.data();
    CHECK(vkCreateDescriptorSetLayout(device, &uniformBufferLayoutInfo, nullptr, descriptorSetLayout));
}

void object::createSkyboxDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout)
{
    std::vector<VkDescriptorSetLayoutBinding> binding;
    binding.push_back(vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    binding.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    VkDescriptorSetLayoutCreateInfo uniformBufferLayoutInfo{};
        uniformBufferLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        uniformBufferLayoutInfo.bindingCount = static_cast<uint32_t>(binding.size());
        uniformBufferLayoutInfo.pBindings = binding.data();
    CHECK(vkCreateDescriptorSetLayout(device, &uniformBufferLayoutInfo, nullptr, descriptorSetLayout));
}
