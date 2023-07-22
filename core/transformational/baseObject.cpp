#include "baseObject.h"
#include "operations.h"
#include "model.h"
#include "dualQuaternion.h"

#include <cstring>

baseObject::baseObject(model* model, uint32_t firstInstance, uint32_t instanceCount) :
    pModel(model),
    firstInstance(firstInstance),
    instanceCount(instanceCount)
{}

void baseObject::destroyUniformBuffers(VkDevice device, std::vector<buffer>& uniformBuffers)
{
    for(auto& buffer: uniformBuffers){
        buffer.destroy(device);
    }
    uniformBuffers.clear();
}

void baseObject::destroy(VkDevice device)
{
    destroyUniformBuffers(device, uniformBuffersHost);
    destroyUniformBuffers(device, uniformBuffersDevice);

    if(descriptorPool )     {vkDestroyDescriptorPool(device, descriptorPool, nullptr); descriptorPool = VK_NULL_HANDLE;}
    if(descriptorSetLayout) {vkDestroyDescriptorSetLayout(device, descriptorSetLayout,  nullptr); descriptorSetLayout = VK_NULL_HANDLE;}
}

void baseObject::updateUniformBuffersFlags(std::vector<buffer>& uniformBuffers)
{
    for (auto& buffer: uniformBuffers){
        buffer.updateFlag = true;
    }
}

uint8_t baseObject::getPipelineBitMask() const
{
    return (outlining.Enable<<4)|(0x0);
}

void baseObject::updateModelMatrix()
{
    dualQuaternion<float> dQuat = convert(rotation,translation);
    glm::mat<4,4,float,glm::defaultp> transformMatrix = convert(dQuat);
    glm::mat<4,4,float,glm::defaultp> scaleMatrix = glm::scale(glm::mat4x4(1.0f),scaling);

    modelMatrix = globalTransformation * transformMatrix * scaleMatrix;

    updateUniformBuffersFlags(uniformBuffersHost);
}

void baseObject::setGlobalTransform(const glm::mat4x4 & transform)
{
    globalTransformation = transform;
    updateModelMatrix();
}

void baseObject::translate(const glm::vec3 & translate)
{
    translation += quaternion<float>(0.0f,translate);
    updateModelMatrix();
}

void baseObject::setPosition(const glm::vec3& translate)
{
    translation = quaternion<float>(0.0f,translate);
    updateModelMatrix();
}

void baseObject::rotate(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    rotation = convert(ang,ax)*rotation;
    updateModelMatrix();
}

void baseObject::rotate(const quaternion<float>& quat)
{
    rotation = quat * rotation;
    updateModelMatrix();
}

void baseObject::scale(const glm::vec3 & scale)
{
    scaling = scale;
    updateModelMatrix();
}

void baseObject::updateAnimation(uint32_t imageNumber)
{
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
      vkMapMemory(device, buffer.memory, 0, sizeof(UniformBuffer), 0, &buffer.map);
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
    }
}

void baseObject::updateUniformBuffer(VkCommandBuffer commandBuffer, uint32_t frameNumber)
{
    if(uniformBuffersHost[frameNumber].updateFlag){
        UniformBuffer ubo{};
            ubo.modelMatrix = modelMatrix;
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
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);
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
    vkAllocateDescriptorSets(device, &allocInfo, descriptors.data());

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

void baseObject::setEnable(const bool& enable) {
    this->enable = enable;
}

void baseObject::setEnableShadow(const bool& enable) {
    this->enableShadow = enable;
}

void baseObject::setModel(model* model, uint32_t firstInstance, uint32_t instanceCount){
    this->pModel = model;
    this->firstInstance = firstInstance;
    this->instanceCount = instanceCount;
}

void baseObject::setConstantColor(const glm::vec4 &color){
    this->constantColor = color;
    updateUniformBuffersFlags(uniformBuffersHost);
}
void baseObject::setColorFactor(const glm::vec4 & color){
    this->colorFactor = color;
    updateUniformBuffersFlags(uniformBuffersHost);
}
void baseObject::setBloomColor(const glm::vec4 & color){
    this->bloomColor = color;
    updateUniformBuffersFlags(uniformBuffersHost);
}
void baseObject::setBloomFactor(const glm::vec4 &color){
    this->bloomFactor = color;
    updateUniformBuffersFlags(uniformBuffersHost);
}

bool baseObject::getEnable() const {return enable;}
bool baseObject::getEnableShadow() const {return enableShadow;}
model* baseObject::getModel() {
    return pModel;
}

uint32_t baseObject::getInstanceNumber(uint32_t imageNumber) const {
    return firstInstance + (instanceCount > imageNumber ? imageNumber : 0);
}

glm::vec4                       baseObject::getConstantColor() const                        {return constantColor;}
glm::vec4                       baseObject::getColorFactor()   const                        {return colorFactor;}

glm::mat4x4                     baseObject::getModelMatrix()   const                        {return modelMatrix;}

VkDescriptorPool                &baseObject::getDescriptorPool()                            {return descriptorPool;}
std::vector<VkDescriptorSet>    &baseObject::getDescriptorSet()                             {return descriptors;}

void                            baseObject::setOutliningEnable(const bool& enable)          {outlining.Enable = enable;}
void                            baseObject::setOutliningWidth(const float& width)           {outlining.Width = width;}
void                            baseObject::setOutliningColor(const vector<float,4>& color) {outlining.Color = color;}

bool                            baseObject::getOutliningEnable() const                      {return outlining.Enable;}
float                           baseObject::getOutliningWidth()  const                      {return outlining.Width;}
vector<float,4>                 baseObject::getOutliningColor()  const                      {return outlining.Color;}

void                            baseObject::setFirstPrimitive(uint32_t firstPrimitive)      {this->firstPrimitive = firstPrimitive;}
void                            baseObject::setPrimitiveCount(uint32_t primitiveCount)      {this->primitiveCount = primitiveCount;}
void                            baseObject::resetPrimitiveCount()                           {primitiveCount=0;}
void                            baseObject::increasePrimitiveCount()                        {primitiveCount++;}

bool                            baseObject::comparePrimitive(uint32_t primitive)            {return primitive>=firstPrimitive&&primitive<firstPrimitive+primitiveCount;}
uint32_t                        baseObject::getFirstPrimitive() const                       {return firstPrimitive;}
uint32_t                        baseObject::getPrimitiveCount() const                       {return primitiveCount;}

skyboxObject::skyboxObject(const std::vector<std::filesystem::path> &TEXTURE_PATH) : baseObject(), texture(new cubeTexture(TEXTURE_PATH)){}

skyboxObject::~skyboxObject(){
    delete texture;
}

void skyboxObject::translate(const glm::vec3 &translate) {
    static_cast<void>(translate);
}

uint8_t skyboxObject::getPipelineBitMask() const {
    return (0<<4)|(0x1);
}

cubeTexture *skyboxObject::getTexture(){
    return texture;
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
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);
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
    vkAllocateDescriptorSets(device, &allocInfo, descriptors.data());

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
