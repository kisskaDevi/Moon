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

BaseObject::~BaseObject(){
    BaseObject::destroy(device);
}

void BaseObject::destroy(VkDevice device)
{
    if(descriptorPool)     {vkDestroyDescriptorPool(device, descriptorPool, nullptr); descriptorPool = VK_NULL_HANDLE;}
    created = false;
}

void BaseObject::updateUniformBuffersFlags(std::vector<moon::utils::Buffer>& uniformBuffers)
{
    for (auto& buffer: uniformBuffers){
        buffer.raiseFlag();
    }
}

void BaseObject::updateModelMatrix()
{
    moon::math::DualQuaternion<float> dQuat = convert(rotation,translation);
    moon::math::Matrix<float,4,4> transformMatrix = convert(dQuat);

    modelMatrix = globalTransformation * transformMatrix * moon::math::scale(scaling);

    updateUniformBuffersFlags(uniformBuffersHost);
}

BaseObject& BaseObject::setGlobalTransform(const moon::math::Matrix<float,4,4> & transform)
{
    globalTransformation = transform;
    updateModelMatrix();
    return *this;
}

BaseObject& BaseObject::translate(const moon::math::Vector<float,3> & translate)
{
    translation += moon::math::Quaternion<float>(0.0f,translate);
    updateModelMatrix();
    return *this;
}

BaseObject& BaseObject::setTranslation(const moon::math::Vector<float,3>& translate)
{
    translation = moon::math::Quaternion<float>(0.0f,translate);
    updateModelMatrix();
    return *this;
}

BaseObject& BaseObject::rotate(const float & ang ,const moon::math::Vector<float,3> & ax)
{
    rotation = convert(ang, moon::math::Vector<float,3>(normalize(ax))) * rotation;
    updateModelMatrix();
    return *this;
}

BaseObject& BaseObject::rotate(const moon::math::Quaternion<float>& quat)
{
    rotation = quat * rotation;
    updateModelMatrix();
    return *this;
}

BaseObject& BaseObject::scale(const moon::math::Vector<float,3> & scale)
{
    scaling = scale;
    updateModelMatrix();
    return *this;
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
    updateUniformBuffersFlags(uniformBuffersHost);
    return *this;
}
BaseObject& BaseObject::setColorFactor(const moon::math::Vector<float,4> & color){
    this->colorFactor = color;
    updateUniformBuffersFlags(uniformBuffersHost);
    return *this;
}
BaseObject& BaseObject::setBloomColor(const moon::math::Vector<float,4> & color){
    this->bloomColor = color;
    updateUniformBuffersFlags(uniformBuffersHost);
    return *this;
}
BaseObject& BaseObject::setBloomFactor(const moon::math::Vector<float,4> &color){
    this->bloomFactor = color;
    updateUniformBuffersFlags(uniformBuffersHost);
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

void BaseObject::createUniformBuffers(VkPhysicalDevice physicalDevice, VkDevice device, uint32_t imageCount)
{
    uniformBuffersHost.resize(imageCount);
    for (auto& buffer: uniformBuffersHost){
        buffer.create(physicalDevice, device, sizeof(UniformBuffer), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        moon::utils::Memory::instance().nameMemory(buffer, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", baseObject::createUniformBuffers, uniformBuffersHost " + std::to_string(&buffer - &uniformBuffersHost[0]));
    }
    uniformBuffersDevice.resize(imageCount);
    for (auto& buffer: uniformBuffersDevice){
        buffer.create(physicalDevice, device, sizeof(UniformBuffer), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
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

void BaseObject::createDescriptorPool(VkDevice device, uint32_t imageCount)
{
    std::vector<VkDescriptorPoolSize> poolSizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(imageCount)}
    };

    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(imageCount);
    CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool));
}

void BaseObject::createDescriptorSet(VkDevice device, uint32_t imageCount)
{
    descriptorSetLayout = moon::interfaces::Object::createDescriptorSetLayout(device);

    std::vector<VkDescriptorSetLayout> layouts(imageCount, descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        allocInfo.pSetLayouts = layouts.data();
    descriptors.resize(imageCount);
    CHECK(vkAllocateDescriptorSets(device, &allocInfo, descriptors.data()));

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
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void BaseObject::create(
    const moon::utils::PhysicalDevice& device,
    VkCommandPool commandPool,
    uint32_t imageCount)
{
    if(!created){
        CHECK_M(device.instance == VK_NULL_HANDLE, std::string("[ deferredGraphics::bindObject ] VkPhysicalDevice is VK_NULL_HANDLE"));
        CHECK_M(device.getLogical() == VK_NULL_HANDLE, std::string("[ deferredGraphics::bindObject ] VkDevice is VK_NULL_HANDLE"));
        CHECK_M(commandPool == VK_NULL_HANDLE, std::string("[ deferredGraphics::bindObject ] VkCommandPool is VK_NULL_HANDLE"));

        createUniformBuffers(device.instance,device.getLogical(),imageCount);
        createDescriptorPool(device.getLogical(),imageCount);
        createDescriptorSet(device.getLogical(),imageCount);
        created = true;
        this->device = device.getLogical();
    }
}

void BaseObject::printStatus() const
{
    std::cout << "translation\t" << translation.im()[0] << '\t' << translation.im()[1] << '\t' << translation.im()[2] << '\n';
    std::cout << "rotation\t" << rotation.re() << '\t' << rotation.im()[0] << '\t' << rotation.im()[1] << '\t' << rotation.im()[2] << '\n';
    std::cout << "scale\t" << scaling[0] << '\t' << scaling[1] << '\t' << scaling[2] << '\n';
}

void SkyboxObject::destroy(VkDevice device){
    BaseObject::destroy(device);
}

SkyboxObject::SkyboxObject(const std::vector<std::filesystem::path> &TEXTURE_PATH) :
    BaseObject(),
    texture(new moon::utils::CubeTexture(TEXTURE_PATH)){
    pipelineBitMask = moon::interfaces::ObjectType::skybox;
}

SkyboxObject::~SkyboxObject(){
    SkyboxObject::destroy(device);
    delete texture;
}

SkyboxObject& SkyboxObject::setMipLevel(float mipLevel){
    texture->setMipLevel(mipLevel);
    return *this;
}

SkyboxObject& SkyboxObject::translate(const moon::math::Vector<float,3> &) {
    return *this;
}

void SkyboxObject::createDescriptorPool(VkDevice device, uint32_t imageCount){
    std::vector<VkDescriptorPoolSize> poolSizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(imageCount)},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(imageCount)}
    };

    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(imageCount);
    CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool));
}

void SkyboxObject::createDescriptorSet(VkDevice device, uint32_t imageCount){
    descriptorSetLayout = moon::interfaces::Object::createSkyboxDescriptorSetLayout(device);

    std::vector<VkDescriptorSetLayout> layouts(imageCount, descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        allocInfo.pSetLayouts = layouts.data();
    descriptors.resize(imageCount);
    CHECK(vkAllocateDescriptorSets(device, &allocInfo, descriptors.data()));

    for (size_t i = 0; i < imageCount; i++){
        VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffersDevice[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBuffer);

        VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = texture->imageView();
            imageInfo.sampler   = texture->sampler();

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
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void SkyboxObject::create(
    const moon::utils::PhysicalDevice& device,
    VkCommandPool commandPool,
    uint32_t imageCount)
{
    if(!created){
        CHECK_M(device.instance == VK_NULL_HANDLE, std::string("[ deferredGraphics::bindObject ] VkPhysicalDevice is VK_NULL_HANDLE"));
        CHECK_M(device.getLogical() == VK_NULL_HANDLE, std::string("[ deferredGraphics::bindObject ] VkDevice is VK_NULL_HANDLE"));
        CHECK_M(commandPool == VK_NULL_HANDLE, std::string("[ deferredGraphics::bindObject ] VkCommandPool is VK_NULL_HANDLE"));

        if(texture){
            VkCommandBuffer commandBuffer = moon::utils::singleCommandBuffer::create(device.getLogical(),commandPool);
            texture->create(device.instance, device.getLogical(), commandBuffer);
            moon::utils::singleCommandBuffer::submit(device.getLogical(),device.getQueue(0,0),commandPool,&commandBuffer);
            texture->destroyCache();
        }
        createUniformBuffers(device.instance,device.getLogical(),imageCount);
        createDescriptorPool(device.getLogical(),imageCount);
        createDescriptorSet(device.getLogical(),imageCount);
        created = true;
        this->device = device.getLogical();
    }
}

}
