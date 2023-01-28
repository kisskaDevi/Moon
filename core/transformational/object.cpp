#include "object.h"
#include "gltfmodel.h"
#include "core/operations.h"
#include "libs/dualQuaternion.h"

#include <iostream>

object::object()
{}

object::object(uint32_t modelCount, gltfModel** model) :
    pModel(model),
    modelCount(modelCount)
{}

object::~object()
{}

void object::destroyUniformBuffers(VkDevice* device)
{
    for(auto& buffer: uniformBuffers){
        if(buffer.instance) vkDestroyBuffer(*device, buffer.instance, nullptr);
        if(buffer.memory)   vkFreeMemory(*device, buffer.memory, nullptr);
    }
    for(auto& buffer: uniformBuffersDevice){
        if(buffer.instance) vkDestroyBuffer(*device, buffer.instance, nullptr);
        if(buffer.memory)   vkFreeMemory(*device, buffer.memory, nullptr);
    }
}

void object::destroy(VkDevice* device)
{
    if(descriptorPool )     vkDestroyDescriptorPool(*device, descriptorPool, nullptr);
    if(descriptorSetLayout) vkDestroyDescriptorSetLayout(*device, descriptorSetLayout,  nullptr);
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

    for (auto& buffer: uniformBuffers){
        buffer.updateFlag = true;
    }
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
    if(modelCount>1){
        if(pModel[imageNumber]->animations.size() > 0){
            if(!changeAnimationFlag){
                if (animationTimer > pModel[imageNumber]->animations[animationIndex].end)
                    animationTimer -= pModel[imageNumber]->animations[animationIndex].end;
                pModel[imageNumber]->updateAnimation(animationIndex, animationTimer);
            }else{
                pModel[imageNumber]->changeAnimation(animationIndex, newAnimationIndex, startTimer, animationTimer, changeAnimationTime);
                if(startTimer+changeAnimationTime<animationTimer){
                    changeAnimationFlag = false;
                    animationTimer = pModel[imageNumber]->animations[animationIndex+1].start;
                    animationIndex = newAnimationIndex;
                }
            }
        }
    }
}

void object::createUniformBuffers(VkPhysicalDevice* physicalDevice, VkDevice* device, uint32_t imageCount)
{
    uniformBuffers.resize(imageCount);
    for (auto& buffer: uniformBuffers){
      Buffer::create(   *physicalDevice,
                        *device,
                        sizeof(UniformBuffer),
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        &buffer.instance,
                        &buffer.memory);
    }
    uniformBuffersDevice.resize(imageCount);
    for (auto& buffer: uniformBuffersDevice){
      Buffer::create(   *physicalDevice,
                        *device,
                        sizeof(UniformBuffer),
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        &buffer.instance,
                        &buffer.memory);
    }
}

void object::updateUniformBuffer(VkDevice device, VkCommandBuffer commandBuffer, uint32_t frameNumber)
{
    if(void* data; uniformBuffers[frameNumber].updateFlag){
        UniformBuffer ubo{};
            ubo.modelMatrix = modelMatrix;
            ubo.constantColor = constantColor;
            ubo.colorFactor = colorFactor;
            ubo.bloomColor = bloomColor;
            ubo.bloomFactor = bloomFactor;
        vkMapMemory(device, uniformBuffers[frameNumber].memory, 0, sizeof(ubo), 0, &data);
            memcpy(data, &ubo, sizeof(ubo));
        vkUnmapMemory(device, uniformBuffers[frameNumber].memory);

        uniformBuffers[frameNumber].updateFlag = false;

        Buffer::copy(commandBuffer, sizeof(UniformBuffer), uniformBuffers[frameNumber].instance, uniformBuffersDevice[frameNumber].instance);
    }
}

void object::createDescriptorPool(VkDevice* device, uint32_t imageCount)
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
    vkCreateDescriptorPool(*device, &poolInfo, nullptr, &descriptorPool);
}

void object::createDescriptorSet(VkDevice* device, uint32_t imageCount)
{
    createObjectDescriptorSetLayout(device,&descriptorSetLayout);

    std::vector<VkDescriptorSetLayout> layouts(imageCount, descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        allocInfo.pSetLayouts = layouts.data();
    descriptors.resize(imageCount);
    vkAllocateDescriptorSets(*device, &allocInfo, descriptors.data());

    for (size_t i = 0; i < imageCount; i++){
        VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffersDevice[i].instance;
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBuffer);

        std::vector<VkWriteDescriptorSet> descriptorWrites{};
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.back().dstBinding = descriptorWrites.size() - 1;
            descriptorWrites.back().dstSet = descriptors[i];
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pBufferInfo = &bufferInfo;
        vkUpdateDescriptorSets(*device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void                            object::setEnable(const bool& enable)                   {this->enable = enable;}
void                            object::setEnableShadow(const bool& enable)             {this->enableShadow = enable;}
void                            object::setModel(gltfModel** model3D)                   {this->pModel = model3D;}

void                            object::setConstantColor(const glm::vec4 &color){
    this->constantColor = color;
    for (auto& buffer: uniformBuffers)
        buffer.updateFlag = true;
}
void                            object::setColorFactor(const glm::vec4 & color){
    this->colorFactor = color;
    for (auto& buffer: uniformBuffers)
        buffer.updateFlag = true;
}
void                            object::setBloomColor(const glm::vec4 & color){
    this->bloomColor = color;
    for (auto& buffer: uniformBuffers)
        buffer.updateFlag = true;
}
void                            object::setBloomFactor(const glm::vec4 &color){
    this->bloomFactor = color;
    for (auto& buffer: uniformBuffers)
        buffer.updateFlag = true;
}

bool                            object::getEnable() const                               {return enable;}
bool                            object::getEnableShadow() const                         {return enableShadow;}
gltfModel*                      object::getModel(uint32_t index)                        {
    gltfModel* model;
    if(modelCount>1&&index>=modelCount){
        model = nullptr;
    }else if(modelCount==1){
        model = pModel[0];
    }else{
        model = pModel[index];
    }
    return model;
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

void skyboxObject::createTexture(VkPhysicalDevice* physicalDevice, VkDevice* device, VkQueue* queue, VkCommandPool* commandPool){
    VkCommandBuffer commandBuffer = SingleCommandBuffer::create(*device,*commandPool);
    texture->createTextureImage(*physicalDevice, *device, commandBuffer);
    SingleCommandBuffer::submit(*device, *queue, *commandPool, &commandBuffer);
    texture->createTextureImageView(device);
    texture->createTextureSampler(device,{VK_FILTER_LINEAR,VK_FILTER_LINEAR,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT});
    texture->destroyStagingBuffer(device);
}

void skyboxObject::destroyTexture(VkDevice* device){
    texture->destroy(device);
}

void skyboxObject::createDescriptorPool(VkDevice* device, uint32_t imageCount){
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
    vkCreateDescriptorPool(*device, &poolInfo, nullptr, &descriptorPool);
}

void skyboxObject::createDescriptorSet(VkDevice* device, uint32_t imageCount){
    createSkyboxObjectDescriptorSetLayout(device,&descriptorSetLayout);

    std::vector<VkDescriptorSetLayout> layouts(imageCount, descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        allocInfo.pSetLayouts = layouts.data();
    descriptors.resize(imageCount);
    vkAllocateDescriptorSets(*device, &allocInfo, descriptors.data());

    for (size_t i = 0; i < imageCount; i++){
        VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i].instance;
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
            descriptorWrites.back().dstBinding = descriptorWrites.size() - 1;
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pBufferInfo = &bufferInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = descriptors[i];
            descriptorWrites.back().dstBinding = descriptorWrites.size() - 1;
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &imageInfo;
        vkUpdateDescriptorSets(*device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}
