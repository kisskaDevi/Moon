#include "baseObject.h"
#include "operations.h"
#include "model.h"
#include "dualQuaternion.h"
#include "device.h"

#include <cstring>

namespace moon::transformational {

BaseObject::BaseObject(moon::interfaces::Model* model, uint32_t firstInstance, uint32_t instanceCount)
{
    pipelineBitMask = moon::interfaces::ObjectType::base | (outlining.Enable ? moon::interfaces::ObjectProperty::outlining : moon::interfaces::ObjectProperty::non);
    pModel = model;
    this->firstInstance = firstInstance;
    this->instanceCount = instanceCount;
}

BaseObject& BaseObject::updateModelMatrix() {
    moon::math::Matrix<float,4,4> transformMatrix = convert(convert(rotation, translation));
    modelMatrix = globalTransformation * transformMatrix * moon::math::scale(scaling);
    utils::raiseFlags(uniformBuffersHost);
    return *this;
}

BaseObject& BaseObject::setGlobalTransform(const moon::math::Matrix<float,4,4> & transform)
{
    globalTransformation = transform;
    return updateModelMatrix();
}

BaseObject& BaseObject::translate(const moon::math::Vector<float,3> & translate)
{
    translation += moon::math::Quaternion<float>(0.0f,translate);
    return updateModelMatrix();
}

BaseObject& BaseObject::setTranslation(const moon::math::Vector<float,3>& translate)
{
    translation = moon::math::Quaternion<float>(0.0f,translate);
    return updateModelMatrix();
}

BaseObject& BaseObject::rotate(const float & ang ,const moon::math::Vector<float,3> & ax)
{
    rotation = convert(ang, moon::math::Vector<float,3>(normalize(ax))) * rotation;
    return updateModelMatrix();
}

BaseObject& BaseObject::rotate(const moon::math::Quaternion<float>& quat)
{
    rotation = quat * rotation;
    return updateModelMatrix();
}

BaseObject& BaseObject::scale(const moon::math::Vector<float,3> & scale)
{
    scaling = scale;
    return updateModelMatrix();
}

const moon::math::Vector<float,3> BaseObject::getTranslation() const{
    return translation.im();
}

const moon::math::Quaternion<float> BaseObject::getRotation() const{
    return rotation;
}

const moon::math::Vector<float,3> BaseObject::getScale() const{
    return scaling;
}

BaseObject& BaseObject::setConstantColor(const moon::math::Vector<float,4> &color){
    this->constantColor = color;
    utils::raiseFlags(uniformBuffersHost);
    return *this;
}
BaseObject& BaseObject::setColorFactor(const moon::math::Vector<float,4> & color){
    this->colorFactor = color;
    utils::raiseFlags(uniformBuffersHost);
    return *this;
}
BaseObject& BaseObject::setBloomColor(const moon::math::Vector<float,4> & color){
    this->bloomColor = color;
    utils::raiseFlags(uniformBuffersHost);
    return *this;
}
BaseObject& BaseObject::setBloomFactor(const moon::math::Vector<float,4> &color){
    this->bloomFactor = color;
    utils::raiseFlags(uniformBuffersHost);
    return *this;
}

void BaseObject::updateAnimation(uint32_t imageNumber, float frameTime){
    animationTimer += frameTime;
    if(uint32_t index = getInstanceNumber(imageNumber); pModel->hasAnimation(index)){
        if(float end = pModel->animationEnd(index, animationIndex); !changeAnimationFlag){
            animationTimer -= animationTimer > end ? end : 0;
            pModel->updateAnimation(index, animationIndex, animationTimer);
        }else{
            pModel->changeAnimation(index, animationIndex, newAnimationIndex, startTimer, animationTimer, changeAnimationTime);
            if(startTimer + changeAnimationTime < animationTimer){
                changeAnimationFlag = false;
                animationIndex = newAnimationIndex;
                animationTimer = pModel->animationStart(index, animationIndex);
            }
        }
    }
}

void BaseObject::changeAnimation(uint32_t newAnimationIndex, float changeAnimationTime){
    changeAnimationFlag = true;
    startTimer = animationTimer;
    this->changeAnimationTime = changeAnimationTime;
    this->newAnimationIndex = newAnimationIndex;
}

void BaseObject::setAnimation(uint32_t animationIndex, float animationTime){
    this->animationIndex = animationIndex;
    this->animationTimer = animationTime;
}

uint32_t BaseObject::getAnimationIndex(){
    return animationIndex;
}

void BaseObject::createUniformBuffers(uint32_t imageCount)
{
    uniformBuffersHost.resize(imageCount);
    for (auto& buffer: uniformBuffersHost){
        buffer = utils::vkDefault::Buffer(device->instance, device->getLogical(), sizeof(UniformBuffer), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        moon::utils::Memory::instance().nameMemory(buffer, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", baseObject::createUniformBuffers, uniformBuffersHost " + std::to_string(&buffer - &uniformBuffersHost[0]));
    }
    uniformBuffersDevice.resize(imageCount);
    for (auto& buffer: uniformBuffersDevice){
        buffer = utils::vkDefault::Buffer(device->instance, device->getLogical(), sizeof(UniformBuffer), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        moon::utils::Memory::instance().nameMemory(buffer, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", baseObject::createUniformBuffers, uniformBuffersDevice " + std::to_string(&buffer - &uniformBuffersDevice[0]));
    }
}

void BaseObject::update(uint32_t frameNumber, VkCommandBuffer commandBuffer)
{
    if(auto& buffer = uniformBuffersHost[frameNumber]; buffer.dropFlag()){
        UniformBuffer ubo{};
            ubo.modelMatrix = transpose(modelMatrix);
            ubo.constantColor = constantColor;
            ubo.colorFactor = colorFactor;
            ubo.bloomColor = bloomColor;
            ubo.bloomFactor = bloomFactor;
        buffer.copy(&ubo);

        moon::utils::buffer::copy(commandBuffer, sizeof(UniformBuffer), buffer, uniformBuffersDevice[frameNumber]);
    }
}

void BaseObject::createDescriptorPool(uint32_t imageCount) {
    descriptorSetLayout = moon::interfaces::Object::createDescriptorSetLayout(device->getLogical());
    descriptorPool = utils::vkDefault::DescriptorPool(device->getLogical(), { &descriptorSetLayout }, imageCount);
    descriptors = descriptorPool.allocateDescriptorSets(descriptorSetLayout, imageCount);
}

void BaseObject::createDescriptorSet(uint32_t imageCount) {
    for (size_t i = 0; i < imageCount; i++){
        VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffersDevice[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBuffer);

        std::vector<VkWriteDescriptorSet> descriptorWrites{};
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstSet = descriptors[i];
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pBufferInfo = &bufferInfo;
        vkUpdateDescriptorSets(device->getLogical(), static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void BaseObject::create(
    const moon::utils::PhysicalDevice& device,
    VkCommandPool commandPool,
    uint32_t imageCount)
{
    if(this->device == VK_NULL_HANDLE){
        CHECK_M(device.instance == VK_NULL_HANDLE, std::string("[ deferredGraphics::bindObject ] VkPhysicalDevice is VK_NULL_HANDLE"));
        CHECK_M(device.getLogical() == VK_NULL_HANDLE, std::string("[ deferredGraphics::bindObject ] VkDevice is VK_NULL_HANDLE"));
        CHECK_M(commandPool == VK_NULL_HANDLE, std::string("[ deferredGraphics::bindObject ] VkCommandPool is VK_NULL_HANDLE"));

        this->device = &device;
        createUniformBuffers(imageCount);
        createDescriptorPool(imageCount);
        createDescriptorSet(imageCount);
    }
}

void BaseObject::printStatus() const {
    std::cout << "translation\t" << translation.im()[0] << '\t' << translation.im()[1] << '\t' << translation.im()[2] << '\n';
    std::cout << "rotation\t" << rotation.re() << '\t' << rotation.im()[0] << '\t' << rotation.im()[1] << '\t' << rotation.im()[2] << '\n';
    std::cout << "scale\t" << scaling[0] << '\t' << scaling[1] << '\t' << scaling[2] << '\n';
}


SkyboxObject::SkyboxObject(const std::vector<std::filesystem::path> &texturePaths) :
    BaseObject(),
    texturePaths(texturePaths){
    pipelineBitMask = moon::interfaces::ObjectType::skybox;
}

SkyboxObject& SkyboxObject::setMipLevel(float mipLevel){
    texture.setMipLevel(mipLevel);
    return *this;
}

SkyboxObject& SkyboxObject::translate(const moon::math::Vector<float,3> &) {
    return *this;
}

void SkyboxObject::createDescriptorPool(uint32_t imageCount) {
    descriptorSetLayout = moon::interfaces::Object::createSkyboxDescriptorSetLayout(device->getLogical());
    descriptorPool = utils::vkDefault::DescriptorPool(device->getLogical(), { &descriptorSetLayout }, imageCount);
    descriptors = descriptorPool.allocateDescriptorSets(descriptorSetLayout, imageCount);
}

void SkyboxObject::createDescriptorSet(uint32_t imageCount) {
    for (size_t i = 0; i < imageCount; i++){
        VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffersDevice[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBuffer);

        VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = texture.imageView();
            imageInfo.sampler = texture.sampler();

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = descriptors[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pBufferInfo = &bufferInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = descriptors[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &imageInfo;
        vkUpdateDescriptorSets(device->getLogical(), static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void SkyboxObject::create(
    const moon::utils::PhysicalDevice& device,
    VkCommandPool commandPool,
    uint32_t imageCount)
{
    if(this->device == VK_NULL_HANDLE){
        CHECK_M(device.instance == VK_NULL_HANDLE, std::string("[ deferredGraphics::bindObject ] VkPhysicalDevice is VK_NULL_HANDLE"));
        CHECK_M(device.getLogical() == VK_NULL_HANDLE, std::string("[ deferredGraphics::bindObject ] VkDevice is VK_NULL_HANDLE"));
        CHECK_M(commandPool == VK_NULL_HANDLE, std::string("[ deferredGraphics::bindObject ] VkCommandPool is VK_NULL_HANDLE"));

        this->device = &device;

        VkCommandBuffer commandBuffer = moon::utils::singleCommandBuffer::create(device.getLogical(), commandPool);
        texture = texturePaths.empty() ? utils::Texture::empty(device, commandBuffer) : utils::CubeTexture(texturePaths, device.instance, device.getLogical(), commandBuffer);
        moon::utils::singleCommandBuffer::submit(device.getLogical(), device.getQueue(0, 0), commandPool, &commandBuffer);
        texture.destroyCache();

        createUniformBuffers(imageCount);
        createDescriptorPool(imageCount);
        createDescriptorSet(imageCount);
    }
}

}
