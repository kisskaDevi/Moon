#include "baseObject.h"
#include "operations.h"
#include "model.h"
#include "dualQuaternion.h"
#include "device.h"

#include <cstring>

baseObject::baseObject(model* model, uint32_t firstInstance, uint32_t instanceCount)
{
    pModel = model;
    this->firstInstance = firstInstance;
    this->instanceCount = instanceCount;
}

void baseObject::destroy(VkDevice device)
{
    destroyBuffers(device, uniformBuffersHost);
    destroyBuffers(device, uniformBuffersDevice);

    if(descriptorPool )     {vkDestroyDescriptorPool(device, descriptorPool, nullptr); descriptorPool = VK_NULL_HANDLE;}
    if(descriptorSetLayout) {vkDestroyDescriptorSetLayout(device, descriptorSetLayout,  nullptr); descriptorSetLayout = VK_NULL_HANDLE;}
    created = false;
}

std::vector<VkDescriptorSet> &baseObject::getDescriptorSet() {
    return descriptors;
}

void baseObject::updateUniformBuffersFlags(std::vector<buffer>& uniformBuffers)
{
    for (auto& buffer: uniformBuffers){
        buffer.updateFlag = true;
    }
}

uint8_t baseObject::getPipelineBitMask() const
{
    return objectType::base | (outlining.Enable ? objectProperty::outlining : objectProperty::non);
}

void baseObject::updateModelMatrix()
{
    dualQuaternion<float> dQuat = convert(rotation,translation);
    matrix<float,4,4> transformMatrix = convert(dQuat);

    modelMatrix = globalTransformation * transformMatrix * ::scale(scaling);

    updateUniformBuffersFlags(uniformBuffersHost);
}

baseObject& baseObject::setGlobalTransform(const matrix<float,4,4> & transform)
{
    globalTransformation = transform;
    updateModelMatrix();
    return *this;
}

baseObject& baseObject::translate(const vector<float,3> & translate)
{
    translation += quaternion<float>(0.0f,translate);
    updateModelMatrix();
    return *this;
}

baseObject& baseObject::setTranslation(const vector<float,3>& translate)
{
    translation = quaternion<float>(0.0f,translate);
    updateModelMatrix();
    return *this;
}

baseObject& baseObject::rotate(const float & ang ,const vector<float,3> & ax)
{
    rotation = convert(ang,vector<float,3>(normalize(ax)))*rotation;
    updateModelMatrix();
    return *this;
}

baseObject& baseObject::rotate(const quaternion<float>& quat)
{
    rotation = quat * rotation;
    updateModelMatrix();
    return *this;
}

baseObject& baseObject::scale(const vector<float,3> & scale)
{
    scaling = scale;
    updateModelMatrix();
    return *this;
}

baseObject& baseObject::setConstantColor(const vector<float,4> &color){
    this->constantColor = color;
    updateUniformBuffersFlags(uniformBuffersHost);
    return *this;
}
baseObject& baseObject::setColorFactor(const vector<float,4> & color){
    this->colorFactor = color;
    updateUniformBuffersFlags(uniformBuffersHost);
    return *this;
}
baseObject& baseObject::setBloomColor(const vector<float,4> & color){
    this->bloomColor = color;
    updateUniformBuffersFlags(uniformBuffersHost);
    return *this;
}
baseObject& baseObject::setBloomFactor(const vector<float,4> &color){
    this->bloomFactor = color;
    updateUniformBuffersFlags(uniformBuffersHost);
    return *this;
}

void baseObject::updateAnimation(uint32_t imageNumber, float frameTime){
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

void baseObject::changeAnimation(uint32_t newAnimationIndex, float changeAnimationTime){
    changeAnimationFlag = true;
    startTimer = animationTimer;
    this->changeAnimationTime = changeAnimationTime;
    this->newAnimationIndex = newAnimationIndex;
}

void baseObject::setAnimation(uint32_t animationIndex, float animationTime){
    this->animationIndex = animationIndex;
    this->animationTimer = animationTime;
}

uint32_t baseObject::getAnimationIndex(){
    return animationIndex;
}

void baseObject::createUniformBuffers(VkPhysicalDevice physicalDevice, VkDevice device, uint32_t imageCount)
{
    uniformBuffersHost.resize(imageCount);
    for (auto& buffer: uniformBuffersHost){
        Buffer::create(   physicalDevice,
                        device,
                        sizeof(UniformBuffer),
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        &buffer.instance,
                        &buffer.memory);
        CHECK(vkMapMemory(device, buffer.memory, 0, sizeof(UniformBuffer), 0, &buffer.map));

        Memory::instance().instance().nameMemory(buffer.memory, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", baseObject::createUniformBuffers, uniformBuffersHost " + std::to_string(&buffer - &uniformBuffersHost[0]));
    }
    uniformBuffersDevice.resize(imageCount);
    for (auto& buffer: uniformBuffersDevice){
        Buffer::create(   physicalDevice,
                        device,
                        sizeof(UniformBuffer),
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        &buffer.instance,
                        &buffer.memory);

        Memory::instance().instance().nameMemory(buffer.memory, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", baseObject::createUniformBuffers, uniformBuffersDevice " + std::to_string(&buffer - &uniformBuffersDevice[0]));
    }
}

void baseObject::update(uint32_t frameNumber, VkCommandBuffer commandBuffer)
{
    if(uniformBuffersHost[frameNumber].updateFlag){
        UniformBuffer ubo{};
            ubo.modelMatrix = transpose(modelMatrix);
            ubo.constantColor = constantColor;
            ubo.colorFactor = colorFactor;
            ubo.bloomColor = bloomColor;
            ubo.bloomFactor = bloomFactor;
        std::memcpy(uniformBuffersHost[frameNumber].map, &ubo, sizeof(ubo));

        uniformBuffersHost[frameNumber].updateFlag = false;

        Buffer::copy(commandBuffer, sizeof(UniformBuffer), uniformBuffersHost[frameNumber].instance, uniformBuffersDevice[frameNumber].instance);
    }
}

void baseObject::createDescriptorPool(VkDevice device, uint32_t imageCount)
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

void baseObject::createDescriptorSet(VkDevice device, uint32_t imageCount)
{
    object::createDescriptorSetLayout(device,&descriptorSetLayout);

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
            bufferInfo.buffer = uniformBuffersDevice[i].instance;
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

void baseObject::create(
    physicalDevice device,
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
    }
}

void baseObject::printStatus() const
{
    std::cout << "translation\t" << translation.im()[0] << '\t' << translation.im()[1] << '\t' << translation.im()[2] << '\n';
    std::cout << "rotation\t" << rotation.re() << '\t' << rotation.im()[0] << '\t' << rotation.im()[1] << '\t' << rotation.im()[2] << '\n';
    std::cout << "scale\t" << scaling[0] << '\t' << scaling[1] << '\t' << scaling[2] << '\n';
}

skyboxObject::skyboxObject(const std::vector<std::filesystem::path> &TEXTURE_PATH) : baseObject(), texture(new cubeTexture(TEXTURE_PATH)){}

skyboxObject::~skyboxObject(){
    delete texture;
}

skyboxObject& skyboxObject::setMipLevel(float mipLevel){
    texture->setMipLevel(mipLevel);
    return *this;
}

skyboxObject& skyboxObject::translate(const vector<float,3> &) {
    return *this;
}

uint8_t skyboxObject::getPipelineBitMask() const {
    return objectType::skybox;
}

void skyboxObject::createDescriptorPool(VkDevice device, uint32_t imageCount){
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

void skyboxObject::createDescriptorSet(VkDevice device, uint32_t imageCount){
    object::createSkyboxDescriptorSetLayout(device,&descriptorSetLayout);

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
            bufferInfo.buffer = uniformBuffersDevice[i].instance;
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBuffer);

        VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = *texture->getTextureImageView();
            imageInfo.sampler   = *texture->getTextureSampler();

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

void skyboxObject::create(
    physicalDevice device,
    VkCommandPool commandPool,
    uint32_t imageCount)
{
    if(!created){
        CHECK_M(device.instance == VK_NULL_HANDLE, std::string("[ deferredGraphics::bindObject ] VkPhysicalDevice is VK_NULL_HANDLE"));
        CHECK_M(device.getLogical() == VK_NULL_HANDLE, std::string("[ deferredGraphics::bindObject ] VkDevice is VK_NULL_HANDLE"));
        CHECK_M(commandPool == VK_NULL_HANDLE, std::string("[ deferredGraphics::bindObject ] VkCommandPool is VK_NULL_HANDLE"));

        if(texture){
            VkCommandBuffer commandBuffer = SingleCommandBuffer::create(device.getLogical(),commandPool);
            texture->createTextureImage(device.instance, device.getLogical(), commandBuffer);
            SingleCommandBuffer::submit(device.getLogical(),device.getQueue(0,0),commandPool,&commandBuffer);
            texture->createTextureImageView(device.getLogical());
            texture->createTextureSampler(device.getLogical(),{VK_FILTER_LINEAR,VK_FILTER_LINEAR,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT});
            texture->destroyStagingBuffer(device.getLogical());
        }
        createUniformBuffers(device.instance,device.getLogical(),imageCount);
        createDescriptorPool(device.getLogical(),imageCount);
        createDescriptorSet(device.getLogical(),imageCount);
        created = true;
    }
}

void skyboxObject::destroy(VkDevice device)
{
    if(texture){
        texture->destroy(device);
    }

    baseObject::destroy(device);
    created = false;
}
