#include "object.h"
#include "gltfmodel.h"
#include "core/operations.h"

#include <iostream>

object::object()
{
    modelMatrix = glm::mat4x4(1.0f);
    m_globalTransform = glm::mat4x4(1.0f);
    m_translate = glm::vec3(0.0f,0.0f,0.0f);
    m_rotate = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_scale = glm::vec3(1.0f,1.0f,1.0f);

}

object::object(uint32_t modelCount, gltfModel** model)
{
    modelMatrix = glm::mat4x4(1.0f);
    m_globalTransform = glm::mat4x4(1.0f);
    m_translate = glm::vec3(0.0f,0.0f,0.0f);
    m_rotate = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_scale = glm::vec3(1.0f,1.0f,1.0f);

    this->pModel = model;
    this->modelCount = modelCount;
}

object::~object()
{

}

void object::destroyUniformBuffers(VkDevice* device)
{
    for(size_t i=0;i<uniformBuffers.size();i++)
    {
        if (uniformBuffers[i] != VK_NULL_HANDLE)
        {
            vkDestroyBuffer(*device, uniformBuffers.at(i), nullptr);
            vkFreeMemory(*device, uniformBuffersMemory.at(i), nullptr);
        }
    }
}

void object::destroy(VkDevice* device)
{
    if (descriptorPool != VK_NULL_HANDLE){
        vkDestroyDescriptorPool(*device, descriptorPool, nullptr);
        descriptorPool = VK_NULL_HANDLE;
    }
    if (descriptorSetLayout != VK_NULL_HANDLE){
        vkDestroyDescriptorSetLayout(*device, descriptorSetLayout,  nullptr);
        descriptorSetLayout = VK_NULL_HANDLE;
    }
}

void object::setGlobalTransform(const glm::mat4x4 & transform)
{
    m_globalTransform = transform;
    updateModelMatrix();
}

void object::translate(const glm::vec3 & translate)
{
    m_translate += translate;
    updateModelMatrix();
}

void object::setPosition(const glm::vec3& translate)
{
    m_translate = translate;
    updateModelMatrix();
}

void object::rotate(const float & ang ,const glm::vec3 & ax)
{
    m_rotate = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax))*m_rotate;
    updateModelMatrix();
}

void object::scale(const glm::vec3 & scale)
{
    m_scale = scale;
    updateModelMatrix();
}

void object::updateModelMatrix()
{
    glm::mat4x4 translateMatrix = glm::translate(glm::mat4x4(1.0f),m_translate);
    glm::mat4x4 rotateMatrix = glm::mat4x4(1.0f);
    if(!(m_rotate.x==0&&m_rotate.y==0&&m_rotate.z==0))
    {
        rotateMatrix = glm::rotate(glm::mat4x4(1.0f),2.0f*glm::acos(m_rotate.w),glm::vec3(m_rotate.x,m_rotate.y,m_rotate.z));
    }
    glm::mat4x4 scaleMatrix = glm::scale(glm::mat4x4(1.0f),m_scale);

    modelMatrix = m_globalTransform * translateMatrix * rotateMatrix * scaleMatrix;
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
        DescriptorPoolSizes.at(index).type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        DescriptorPoolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);

    //Мы будем выделять один из этих дескрипторов для каждого кадра. На эту структуру размера пула ссылается главный VkDescriptorPoolCreateInfo:
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(DescriptorPoolSizes.size());
    poolInfo.pPoolSizes = DescriptorPoolSizes.data();
    poolInfo.maxSets = static_cast<uint32_t>(imageCount);

    if (vkCreateDescriptorPool(*device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create object descriptor pool!");
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
    if (vkAllocateDescriptorSets(*device, &allocInfo, descriptors.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate object descriptor sets!");

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

bool                            object::getEnable()             const                   {return enable;}
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
glm::vec4                       object::getConstantColor()      const                   {return constantColor;}
glm::vec4                       object::getColorFactor()        const                   {return colorFactor;}

glm::mat4x4                     &object::ModelMatrix()                                  {return modelMatrix;}
glm::mat4x4                     &object::Transformation()                               {return m_globalTransform;}
glm::vec3                       &object::Translate()                                    {return m_translate;}
glm::quat                       &object::Rotate()                                       {return m_rotate;}
glm::vec3                       &object::Scale()                                        {return m_scale;}

VkDescriptorPool                &object::getDescriptorPool()                            {return descriptorPool;}
std::vector<VkDescriptorSet>    &object::getDescriptorSet()                             {return descriptors;}
std::vector<VkBuffer>           &object::getUniformBuffers()                            {return uniformBuffers;}

void                            object::setOutliningEnable(const bool& enable)            {outlining.Enable = enable;}
void                            object::setOutliningWidth(const float& width)             {outlining.Width = width;}
void                            object::setOutliningColor(const glm::vec4& color)         {outlining.Color = color;}

bool                            object::getOutliningEnable() const                        {return outlining.Enable;}
float                           object::getOutliningWidth()  const                        {return outlining.Width;}
glm::vec4                       object::getOutliningColor()  const                        {return outlining.Color;}

void                            object::setFirstPrimitive(uint32_t firstPrimitive)      {this->firstPrimitive = firstPrimitive;}
void                            object::setPrimitiveCount(uint32_t primitiveCount)      {this->primitiveCount = primitiveCount;}
void                            object::resetPrimitiveCount()                           {primitiveCount=0;}
void                            object::increasePrimitiveCount()                        {primitiveCount++;}

bool                            object::comparePrimitive(uint32_t primitive)            {return primitive>=firstPrimitive&&primitive<firstPrimitive+primitiveCount;}
uint32_t                        object::getFirstPrimitive() const                       {return firstPrimitive;}
uint32_t                        object::getPrimitiveCount() const                       {return primitiveCount;}
