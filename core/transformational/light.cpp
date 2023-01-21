#include "light.h"
#include "object.h"
#include "core/operations.h"
#include "core/graphics/deferredGraphics/renderStages/shadowGraphics.h"
#include "core/texture.h"
#include "libs/dualQuaternion.h"

#include <iostream>

spotLight::spotLight(bool enableShadow, bool enableScattering, uint32_t type):
    enableShadow(enableShadow),
    enableScattering(enableScattering),
    type(type)
{}

spotLight::spotLight(const std::string & TEXTURE_PATH, bool enableShadow, bool enableScattering, uint32_t type):
    tex(new texture(TEXTURE_PATH)),
    enableShadow(enableShadow),
    enableScattering(enableScattering),
    type(type)
{}

spotLight::~spotLight(){
    delete tex;
}

void spotLight::destroyUniformBuffers(VkDevice* device)
{
    for(auto& buffer: uniformBuffers){
        if(buffer.instance) vkDestroyBuffer(*device, buffer.instance, nullptr);
        if(buffer.memory)   vkFreeMemory(*device, buffer.memory, nullptr);
    }
    uniformBuffers.resize(0);
}

void spotLight::destroy(VkDevice* device)
{
    if(descriptorSetLayout) {vkDestroyDescriptorSetLayout(*device, descriptorSetLayout,  nullptr); descriptorSetLayout = VK_NULL_HANDLE;}
    if(descriptorPool)      {vkDestroyDescriptorPool(*device, descriptorPool, nullptr); descriptorPool = VK_NULL_HANDLE;}

    if(shadow){
        shadow->destroy();
    }
}

void spotLight::updateModelMatrix()
{
    dualQuaternion<float> dQuat = convert(rotation,translation);
    glm::mat<4,4,float,glm::defaultp> transformMatrix = convert(dQuat);
    glm::mat<4,4,float,glm::defaultp> scaleMatrix = glm::scale(glm::mat4x4(1.0f),scaling);

    modelMatrix = globalTransformation * transformMatrix * scaleMatrix;

    for (auto& buffer: uniformBuffers){
        buffer.updateFlag = true;
    }
}

void spotLight::setGlobalTransform(const glm::mat4 & transform)
{
    globalTransformation = transform;
    updateModelMatrix();
}

void spotLight::translate(const glm::vec3 & translate)
{
    translation += quaternion<float>(0.0f,translate);
    updateModelMatrix();
}

void spotLight::rotate(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    rotation = convert(ang,ax)*rotation;
    updateModelMatrix();
}

void spotLight::scale(const glm::vec3 & scale)
{
    scaling = scale;
    updateModelMatrix();
}

void spotLight::setPosition(const glm::vec3& translate)
{
    translation = quaternion<float>(0.0f,translate);
    updateModelMatrix();
}

void spotLight::rotateX(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    rotationX = convert(ang,ax) * rotationX;
    rotation = rotationX * rotationY;
    updateModelMatrix();
}

void spotLight::rotateY(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    rotationY = convert(ang,ax) * rotationY;
    rotation = rotationX * rotationY;
    updateModelMatrix();
}

void                            spotLight::setShadowExtent(const VkExtent2D & shadowExtent)     {this->shadowExtent = shadowExtent;}
void                            spotLight::setShadow(bool enable)                               {enableShadow = enable;}
void                            spotLight::setScattering(bool enable)                           {enableScattering = enable;}
void                            spotLight::setTexture(texture* tex)                             {this->tex = tex;}
void                            spotLight::setProjectionMatrix(const glm::mat4x4 & projection)  {
    projectionMatrix = projection;
    for (auto& buffer: uniformBuffers){
        buffer.updateFlag = true;
    }
}
void                            spotLight::setLightColor(const glm::vec4 &color){
    lightColor = color;
    for (auto& buffer: uniformBuffers){
        buffer.updateFlag = true;
    }
}

glm::mat4x4                     spotLight::getModelMatrix() const {return modelMatrix;}
glm::vec3                       spotLight::getTranslate() const {return translation.vector();;}
glm::vec4                       spotLight::getLightColor() const {return lightColor;}
texture*                        spotLight::getTexture(){return tex;}

uint8_t                         spotLight::getPipelineBitMask()     {return (enableScattering<<5)|(isShadowEnable()<<4)|(0x0);}

bool                            spotLight::isShadowEnable() const{return shadow ? true : false;}
bool                            spotLight::isScatteringEnable() const{return enableScattering;}

VkDescriptorSet*                spotLight::getDescriptorSets(){return descriptorSets.data();}
VkCommandBuffer*                spotLight::getShadowCommandBuffer(uint32_t imageCount) {return shadow->getCommandBuffer(imageCount);}

void spotLight::createUniformBuffers(VkPhysicalDevice* physicalDevice, VkDevice* device, uint32_t imageCount)
{
    uniformBuffers.resize(imageCount);
    for (auto& buffer: uniformBuffers){
        createBuffer(   physicalDevice,
                        device,
                        sizeof(LightBufferObject),
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        buffer.instance,
                        buffer.memory);
    }
}

void spotLight::updateUniformBuffer(VkDevice* device, uint32_t frameNumber)
{
    if(void* data; uniformBuffers[frameNumber].updateFlag){
        LightBufferObject buffer{};
            buffer.proj = projectionMatrix;
            buffer.view = glm::inverse(modelMatrix);
            buffer.projView = projectionMatrix * buffer.view;
            buffer.position = modelMatrix * glm::vec4(0.0f,0.0f,0.0f,1.0f);
            buffer.lightColor = lightColor;
            buffer.lightProp = glm::vec4(static_cast<float>(type),lightPowerFactor,lightDropFactor,0.0f);
        vkMapMemory(*device, uniformBuffers[frameNumber].memory, 0, sizeof(buffer), 0, &data);
            memcpy(data, &buffer, sizeof(buffer));
        vkUnmapMemory(*device, uniformBuffers[frameNumber].memory);

        uniformBuffers[frameNumber].updateFlag = false;
    }
}

void spotLight::createShadow(VkPhysicalDevice* physicalDevice, VkDevice* device, QueueFamilyIndices* queueFamilyIndices, uint32_t imageCount, const std::string& ExternalPath)
{
    if(enableShadow){
        shadow = new shadowGraphics(imageCount,shadowExtent);
        shadow->setExternalPath(ExternalPath);
        shadow->setDeviceProp(physicalDevice,device,queueFamilyIndices);
        shadow->createShadow();
    }
}
void spotLight::updateShadowDescriptorSets()
{
    std::vector<VkBuffer> buffers;
    for(auto& buffer: uniformBuffers)
        buffers.push_back(buffer.instance);

    if(shadow){
        shadow->updateDescriptorSets(buffers.size(),buffers.data(),sizeof(LightBufferObject));
    }
}
void spotLight::createShadowCommandBuffers()
{
    if(shadow){
        shadow->createCommandBuffers();
    }
}
void spotLight::updateShadowCommandBuffer(uint32_t frameNumber, std::vector<object*>& objects)
{
    if(shadow){
        shadow->updateCommandBuffer(frameNumber,objects);
    }
}

void spotLight::createDescriptorPool(VkDevice* device, uint32_t imageCount)
{
    uint32_t index = 0;
    std::array<VkDescriptorPoolSize,3> poolSizes{};
        poolSizes[index] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(imageCount)};
    index++;
        poolSizes[index] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(imageCount)};
    index++;
        poolSizes[index] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(imageCount)};
    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(imageCount);
    vkCreateDescriptorPool(*device, &poolInfo, nullptr, &descriptorPool);
}

void spotLight::createDescriptorSets(VkDevice* device, uint32_t imageCount)
{
    createSpotLightDescriptorSetLayout(device,&descriptorSetLayout);

    descriptorSets.resize(imageCount);
    std::vector<VkDescriptorSetLayout> layouts(imageCount, descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        allocInfo.pSetLayouts = layouts.data();
    vkAllocateDescriptorSets(*device, &allocInfo, descriptorSets.data());
}

void spotLight::updateDescriptorSets(VkDevice* device, uint32_t imageCount, texture* emptyTexture)
{
    for (size_t i=0; i<imageCount; i++)
    {
        uint32_t index = 0;

        VkDescriptorBufferInfo lightBufferInfo{};
            lightBufferInfo.buffer = uniformBuffers[i].instance;
            lightBufferInfo.offset = 0;
            lightBufferInfo.range = sizeof(LightBufferObject);
        VkDescriptorImageInfo shadowImageInfo{};
            shadowImageInfo.imageLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            shadowImageInfo.imageView       = isShadowEnable() ? shadow->getAttachment()->imageView : *emptyTexture->getTextureImageView();
            shadowImageInfo.sampler         = isShadowEnable() ? shadow->getAttachment()->sampler : *emptyTexture->getTextureSampler();
        VkDescriptorImageInfo lightTexture{};
            lightTexture.imageLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            lightTexture.imageView          = tex ? *tex->getTextureImageView() : *emptyTexture->getTextureImageView();
            lightTexture.sampler            = tex ? *tex->getTextureSampler() : *emptyTexture->getTextureSampler();

        index = 0;
        std::array<VkWriteDescriptorSet,3> descriptorWrites{};
            descriptorWrites[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[index].dstSet = descriptorSets[i];
            descriptorWrites[index].dstBinding = index;
            descriptorWrites[index].dstArrayElement = 0;
            descriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[index].descriptorCount = 1;
            descriptorWrites[index].pBufferInfo = &lightBufferInfo;
        index++;
            descriptorWrites[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[index].dstSet = descriptorSets[i];
            descriptorWrites[index].dstBinding = index;
            descriptorWrites[index].dstArrayElement = 0;
            descriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[index].descriptorCount = 1;
            descriptorWrites[index].pImageInfo = &shadowImageInfo;
        index++;
            descriptorWrites[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[index].dstSet = descriptorSets[i];
            descriptorWrites[index].dstBinding = index;
            descriptorWrites[index].dstArrayElement = 0;
            descriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[index].descriptorCount = 1;
            descriptorWrites[index].pImageInfo = &lightTexture;
        vkUpdateDescriptorSets(*device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

//isotropicLight

isotropicLight::isotropicLight(std::vector<spotLight *>& lightSource)
{
    uint32_t number = lightSource.size();
    uint32_t index = number;
    lightSource.push_back(new spotLight(true,false,spotType::square));
    lightSource[index]->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    lightSource[index]->setLightColor(glm::vec4(1.0f,0.0f,0.0f,1.0f));

    index++;
    lightSource.push_back(new spotLight(true,false,spotType::square));
    lightSource[index]->rotate(glm::radians(-90.0f),glm::vec3(1.0f,0.0f,0.0f));
    lightSource[index]->setLightColor(glm::vec4(0.0f,1.0f,0.0f,1.0f));

    index++;
    lightSource.push_back(new spotLight(true,false,spotType::square));
    lightSource[index]->setLightColor(glm::vec4(0.0f,0.0f,1.0f,1.0f));

    index++;
    lightSource.push_back(new spotLight(true,false,spotType::square));
    lightSource[index]->rotate(glm::radians(90.0f),glm::vec3(0.0f,1.0f,0.0f));
    lightSource[index]->setLightColor(glm::vec4(0.3f,0.6f,0.9f,1.0f));

    index++;
    lightSource.push_back(new spotLight(true,false,spotType::square));
    lightSource[index]->rotate(glm::radians(-90.0f),glm::vec3(0.0f,1.0f,0.0f));
    lightSource[index]->setLightColor(glm::vec4(0.6f,0.9f,0.3f,1.0f));

    index++;
    lightSource.push_back(new spotLight(true,false,spotType::square));
    lightSource[index]->rotate(glm::radians(180.0f),glm::vec3(1.0f,0.0f,0.0f));
    lightSource[index]->setLightColor(glm::vec4(0.9f,0.3f,0.6f,1.0f));

    this->lightSource.resize(6);
    for(uint32_t i=0;i<6;i++){
        this->lightSource[i] = lightSource[number+i];
    }
}

isotropicLight::~isotropicLight(){}

glm::vec4       isotropicLight::getLightColor() const {return lightColor;}
glm::vec3       isotropicLight::getTranslate() const {return translation.vector();}

void isotropicLight::setProjectionMatrix(const glm::mat4x4 & projection)
{
    projectionMatrix = projection;
    for(uint32_t i=0;i<6;i++)
        lightSource.at(i)->setProjectionMatrix(projectionMatrix);
}

void isotropicLight::setLightColor(const glm::vec4 &color)
{
    this->lightColor = color;
    for(uint32_t i=0;i<6;i++)
        lightSource.at(i)->setLightColor(color);
}

void isotropicLight::updateModelMatrix()
{
    dualQuaternion<float> dQuat = convert(rotation,translation);
    glm::mat<4,4,float,glm::defaultp> transformMatrix = convert(dQuat);
    glm::mat<4,4,float,glm::defaultp> scaleMatrix = glm::scale(glm::mat4x4(1.0f),scaling);

    modelMatrix = globalTransformation * transformMatrix * scaleMatrix;
}

void isotropicLight::setGlobalTransform(const glm::mat4 & transform)
{
    globalTransformation = transform;
    updateModelMatrix();

    for(uint32_t i=0;i<6;i++)
        lightSource.at(i)->setGlobalTransform(modelMatrix);
}

void isotropicLight::translate(const glm::vec3 & translate)
{
    translation += quaternion<float>(0.0f,-translate);
    updateModelMatrix();
}

void isotropicLight::rotate(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    rotation = convert(ang,ax)*rotation;
    updateModelMatrix();

    for(uint32_t i=0;i<6;i++)
        lightSource.at(i)->setGlobalTransform(modelMatrix);
}

void isotropicLight::scale(const glm::vec3 & scale)
{
    scaling = scale;
    updateModelMatrix();

    for(uint32_t i=0;i<6;i++)
        lightSource.at(i)->setGlobalTransform(modelMatrix);
}

void isotropicLight::rotateX(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    rotationX = convert(ang,ax) * rotationX;
    rotation = rotationX * rotationY;
    updateModelMatrix();

    for(uint32_t i=0;i<6;i++)
        lightSource.at(i)->setGlobalTransform(modelMatrix);
}

void isotropicLight::rotateY(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    rotationY = convert(ang,ax) * rotationY;
    rotation = rotationX * rotationY;
    updateModelMatrix();

    for(uint32_t i=0;i<6;i++)
        lightSource.at(i)->setGlobalTransform(modelMatrix);
}

void isotropicLight::setPosition(const glm::vec3& translate)
{
    translation = quaternion<float>(0.0f,translate);
    updateModelMatrix();

    for(uint32_t i=0;i<6;i++)
        lightSource.at(i)->setGlobalTransform(modelMatrix);
}


