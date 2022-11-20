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
{

}

void object::destroyUniformBuffers(VkDevice* device)
{
    for(size_t i=0;i<uniformBuffers.size();i++)
    {
        if(uniformBuffers[i])       vkDestroyBuffer(*device, uniformBuffers[i], nullptr);
        if(uniformBuffersMemory[i]) vkFreeMemory(*device, uniformBuffersMemory[i], nullptr);
    }
}

void object::destroy(VkDevice* device)
{
    if(descriptorPool )     vkDestroyDescriptorPool(*device, descriptorPool, nullptr);
    if(descriptorSetLayout) vkDestroyDescriptorSetLayout(*device, descriptorSetLayout,  nullptr);
}

void object::updateModelMatrix()
{
    dualQuaternion<float> dQuat = convert(rotation,translation);
    glm::mat<4,4,float,glm::defaultp> transformMatrix = convert(dQuat);
    glm::mat<4,4,float,glm::defaultp> scaleMatrix = glm::scale(glm::mat4x4(1.0f),scaling);

    modelMatrix = globalTransformation * transformMatrix * scaleMatrix;
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
    uniformBuffersMemory.resize(imageCount);
    for (size_t i = 0; i < imageCount; i++){
        createBuffer(   physicalDevice,
                        device,
                        sizeof(UniformBuffer),
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        uniformBuffers[i],
                        uniformBuffersMemory[i]);
    }
}

void object::updateUniformBuffer(VkDevice* device, uint32_t currentImage)
{
    void* data;
    UniformBuffer ubo{};
        ubo.modelMatrix = modelMatrix;
        ubo.constantColor = constantColor;
        ubo.colorFactor = colorFactor;
        ubo.bloomColor = bloomColor;
        ubo.bloomFactor = bloomFactor;
    vkMapMemory(*device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
        memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(*device, uniformBuffersMemory[currentImage]);
}

void object::createDescriptorPool(VkDevice* device, uint32_t imageCount)
{
    size_t index = 0;
    std::vector<VkDescriptorPoolSize> DescriptorPoolSizes(1);
        DescriptorPoolSizes[index].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        DescriptorPoolSizes[index].descriptorCount = static_cast<uint32_t>(imageCount);

    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(DescriptorPoolSizes.size());
        poolInfo.pPoolSizes = DescriptorPoolSizes.data();
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

    for (size_t i = 0; i < imageCount; i++)
    {
        VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBuffer);
        VkWriteDescriptorSet writeDescriptorSet{};
            writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writeDescriptorSet.descriptorCount = 1;
            writeDescriptorSet.dstSet = descriptors[i];
            writeDescriptorSet.dstBinding = 0;
            writeDescriptorSet.pBufferInfo = &bufferInfo;
        vkUpdateDescriptorSets(*device, 1, &writeDescriptorSet, 0, nullptr);
    }
}

void                            object::setEnable(const bool& enable)                   {this->enable = enable;}
void                            object::setModel(gltfModel** model3D)                   {this->pModel = model3D;}
void                            object::setConstantColor(const glm::vec4 &color)        {this->constantColor = color;}
void                            object::setColorFactor(const glm::vec4 & color)         {this->colorFactor = color;}
void                            object::setBloomColor(const glm::vec4 & color)          {this->bloomColor = color;}
void                            object::setBloomFactor(const glm::vec4 &color)          {this->bloomFactor = color;}

bool                            object::getEnable() const                               {return enable;}
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

glm::mat4x4&                    object::ModelMatrix()                                   {return modelMatrix;}

VkDescriptorPool                &object::getDescriptorPool()                            {return descriptorPool;}
std::vector<VkDescriptorSet>    &object::getDescriptorSet()                             {return descriptors;}
std::vector<VkBuffer>           &object::getUniformBuffers()                            {return uniformBuffers;}

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
