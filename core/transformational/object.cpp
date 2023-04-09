#include "object.h"
#include "../utils/operations.h"
#include "../interfaces/model.h"
#include "dualQuaternion.h"

#include <cstring>

object::object()
{}

object::object(model* model, uint32_t firstInstance, uint32_t instanceCount) :
    pModel(model),
    firstInstance(firstInstance),
    instanceCount(instanceCount)
{}

object::~object()
{}

void object::destroyUniformBuffers(VkDevice device, std::vector<buffer>& uniformBuffers)
{
    for(auto& buffer: uniformBuffers){
        if(buffer.map){      vkUnmapMemory(device, buffer.memory); buffer.map = nullptr;}
        if(buffer.instance){ vkDestroyBuffer(device, buffer.instance, nullptr); buffer.instance = VK_NULL_HANDLE;}
        if(buffer.memory){   vkFreeMemory(device, buffer.memory, nullptr); buffer.memory = VK_NULL_HANDLE;}
    }
    uniformBuffers.resize(0);
}

void object::destroy(VkDevice device)
{
    destroyUniformBuffers(device, uniformBuffersHost);
    destroyUniformBuffers(device, uniformBuffersDevice);

    if(descriptorPool )     vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    if(descriptorSetLayout) vkDestroyDescriptorSetLayout(device, descriptorSetLayout,  nullptr);
}

void object::updateUniformBuffersFlags(std::vector<buffer>& uniformBuffers)
{
    for (auto& buffer: uniformBuffers){
        buffer.updateFlag = true;
    }
}

uint8_t object::getPipelineBitMask() const
{
    return (outlining.Enable<<0);
}

void object::updateModelMatrix()
{
    dualQuaternion<float> dQuat = convert(rotation,translation);
    glm::mat<4,4,float,glm::defaultp> transformMatrix = convert(dQuat);
    glm::mat<4,4,float,glm::defaultp> scaleMatrix = glm::scale(glm::mat4x4(1.0f),scaling);

    modelMatrix = globalTransformation * transformMatrix * scaleMatrix;

    updateUniformBuffersFlags(uniformBuffersHost);
}

void object::setGlobalTransform(const glm::mat4x4 & transform)
{
    globalTransformation = transform;
    updateModelMatrix();
}

void object::translate(const glm::vec3 & translate)
{
    translation += quaternion<float>(0.0f,translate);
    updateModelMatrix();
}

void object::setPosition(const glm::vec3& translate)
{
    translation = quaternion<float>(0.0f,translate);
    updateModelMatrix();
}

void object::rotate(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    rotation = convert(ang,ax)*rotation;
    updateModelMatrix();
}

void object::scale(const glm::vec3 & scale)
{
    scaling = scale;
    updateModelMatrix();
}

void object::updateAnimation(uint32_t imageNumber)
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

void object::createUniformBuffers(VkPhysicalDevice physicalDevice, VkDevice device, uint32_t imageCount)
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

void object::updateUniformBuffer(VkCommandBuffer commandBuffer, uint32_t frameNumber)
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

void object::createDescriptorPool(VkDevice device, uint32_t imageCount)
{
    std::vector<VkDescriptorPoolSize> poolSizes;
    poolSizes.push_back(VkDescriptorPoolSize{});
        poolSizes.back().type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes.back().descriptorCount = static_cast<uint32_t>(imageCount);

    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(imageCount);
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);
}

void object::createDescriptorSet(VkDevice device, uint32_t imageCount)
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

void object::setConstantColor(const glm::vec4 &color){
    this->constantColor = color;
    updateUniformBuffersFlags(uniformBuffersHost);
}
void object::setColorFactor(const glm::vec4 & color){
    this->colorFactor = color;
    updateUniformBuffersFlags(uniformBuffersHost);
}
void object::setBloomColor(const glm::vec4 & color){
    this->bloomColor = color;
    updateUniformBuffersFlags(uniformBuffersHost);
}
void object::setBloomFactor(const glm::vec4 &color){
    this->bloomFactor = color;
    updateUniformBuffersFlags(uniformBuffersHost);
}

bool                            object::getEnable() const                               {return enable;}
bool                            object::getEnableShadow() const                         {return enableShadow;}
model*                          object::getModel()                                      {
    return pModel;
}

uint32_t object::getInstanceNumber(uint32_t imageNumber) const {
    return firstInstance + (instanceCount > imageNumber ? imageNumber : 0);
}
glm::vec4                       object::getConstantColor() const                        {return constantColor;}
glm::vec4                       object::getColorFactor()   const                        {return colorFactor;}

glm::mat4x4                     object::getModelMatrix()   const                        {return modelMatrix;}

VkDescriptorPool                &object::getDescriptorPool()                            {return descriptorPool;}
std::vector<VkDescriptorSet>    &object::getDescriptorSet()                             {return descriptors;}

void                            object::setOutliningEnable(const bool& enable)          {outlining.Enable = enable;}
void                            object::setOutliningWidth(const float& width)           {outlining.Width = width;}
void                            object::setOutliningColor(const glm::vec4& color)       {outlining.Color = color;}

bool                            object::getOutliningEnable() const                      {return outlining.Enable;}
float                           object::getOutliningWidth()  const                      {return outlining.Width;}
glm::vec4                       object::getOutliningColor()  const                      {return outlining.Color;}

void                            object::setFirstPrimitive(uint32_t firstPrimitive)      {this->firstPrimitive = firstPrimitive;}
void                            object::setPrimitiveCount(uint32_t primitiveCount)      {this->primitiveCount = primitiveCount;}
void                            object::resetPrimitiveCount()                           {primitiveCount=0;}
void                            object::increasePrimitiveCount()                        {primitiveCount++;}

bool                            object::comparePrimitive(uint32_t primitive)            {return primitive>=firstPrimitive&&primitive<firstPrimitive+primitiveCount;}
uint32_t                        object::getFirstPrimitive() const                       {return firstPrimitive;}
uint32_t                        object::getPrimitiveCount() const                       {return primitiveCount;}

skyboxObject::skyboxObject(const std::vector<std::string> &TEXTURE_PATH) : object(), texture(new cubeTexture(TEXTURE_PATH)){}

skyboxObject::~skyboxObject(){
    delete texture;
}

void skyboxObject::translate(const glm::vec3 &translate)
{
    static_cast<void>(translate);
}

cubeTexture *skyboxObject::getTexture(){
    return texture;
}

void skyboxObject::createDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout)
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

void skyboxObject::createDescriptorPool(VkDevice device, uint32_t imageCount){
    std::vector<VkDescriptorPoolSize> poolSizes;
    poolSizes.push_back(VkDescriptorPoolSize{});
        poolSizes.back().type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes.back().descriptorCount = static_cast<uint32_t>(imageCount);
    poolSizes.push_back(VkDescriptorPoolSize{});
        poolSizes.back().type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes.back().descriptorCount = static_cast<uint32_t>(imageCount);

    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(imageCount);
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);
}

void skyboxObject::createDescriptorSet(VkDevice device, uint32_t imageCount){
    skyboxObject::createDescriptorSetLayout(device,&descriptorSetLayout);

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
