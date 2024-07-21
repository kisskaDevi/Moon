#include "object.h"
#include "vkdefault.h"
#include "operations.h"

namespace moon::interfaces {

bool Object::getEnable() const {
    return enable;
}

bool Object::getEnableShadow() const {
    return enableShadow;
}

void Object::setEnable(const bool& inenable) {
    enable = inenable;
}

void Object::setEnableShadow(const bool& inenable) {
    enableShadow = inenable;
}

void Object::setModel(Model* model, uint32_t infirstInstance, uint32_t ininstanceCount){
    pModel = model;
    firstInstance = infirstInstance;
    instanceCount = ininstanceCount;
}

Model* Object::getModel() {
    return pModel;
}

uint32_t Object::getInstanceNumber(uint32_t imageNumber) const {
    return firstInstance + (instanceCount > imageNumber ? imageNumber : 0);
}

void Object::setOutlining(const bool& enable, const float& width, const moon::math::Vector<float,4>& color){
    outlining.Enable = enable;
    outlining.Width = width > 0.0f ? width : outlining.Width;
    outlining.Color = dot(color,color) > 0.0f ? color : outlining.Color;

    pipelineBitMask |= outlining.Enable ? ObjectProperty::outlining : ObjectProperty::non;
}

bool Object::getOutliningEnable() const {
    return outlining.Enable;
}

float Object::getOutliningWidth() const {
    return outlining.Width;
}

moon::math::Vector<float,4> Object::getOutliningColor() const {
    return outlining.Color;
}

void Object::setFirstPrimitive(uint32_t infirstPrimitive) {
    firstPrimitive = infirstPrimitive;
}

void Object::setPrimitiveCount(uint32_t inprimitiveCount) {
    primitiveCount = inprimitiveCount;
}

bool Object::comparePrimitive(uint32_t primitive) {
    return !(primitive < firstPrimitive) && (primitive < firstPrimitive + primitiveCount);
}

uint32_t Object::getFirstPrimitive() const {
    return firstPrimitive;
}

uint32_t Object::getPrimitiveCount() const {
    return primitiveCount;
}

const VkDescriptorSet& Object::getDescriptorSet(uint32_t i) const {
    return descriptors[i];
}

uint8_t Object::getPipelineBitMask() const {
    return pipelineBitMask;
}

moon::utils::vkDefault::DescriptorSetLayout Object::createDescriptorSetLayout(VkDevice device){
    std::vector<VkDescriptorSetLayoutBinding> binding;
        binding.push_back(moon::utils::vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
        binding.back().stageFlags |= VK_SHADER_STAGE_FRAGMENT_BIT;
    return utils::vkDefault::DescriptorSetLayout(device, binding);
}

moon::utils::vkDefault::DescriptorSetLayout Object::createSkyboxDescriptorSetLayout(VkDevice device) {
    std::vector<VkDescriptorSetLayoutBinding> binding;
        binding.push_back(moon::utils::vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
        binding.push_back(moon::utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    return utils::vkDefault::DescriptorSetLayout(device, binding);
}

}
