#include "spotLight.h"
#include "../utils/operations.h"
#include "../utils/texture.h"
#include "dualQuaternion.h"

#include <iostream>
#include <cstring>

spotLight::spotLight(bool enableShadow, bool enableScattering, uint32_t type):
    enableShadow(enableShadow),
    enableScattering(enableScattering),
    type(type)
{
    if(enableShadow){
        shadow = new attachments;
    }
}

spotLight::spotLight(const std::string & TEXTURE_PATH, bool enableShadow, bool enableScattering, uint32_t type):
    tex(new texture(TEXTURE_PATH)),
    enableShadow(enableShadow),
    enableScattering(enableScattering),
    type(type)
{
    if(enableShadow){
        shadow = new attachments;
    }
}

spotLight::~spotLight(){
    delete tex;
    if(enableShadow){
        delete shadow;
    }
}

void spotLight::destroyUniformBuffers(VkDevice* device, std::vector<buffer>& uniformBuffers)
{
    for(auto& buffer: uniformBuffers){
        if(buffer.map){      vkUnmapMemory(*device, buffer.memory); buffer.map = nullptr;}
        if(buffer.instance){ vkDestroyBuffer(*device, buffer.instance, nullptr); buffer.instance = VK_NULL_HANDLE;}
        if(buffer.memory){   vkFreeMemory(*device, buffer.memory, nullptr); buffer.memory = VK_NULL_HANDLE;}
    }
    uniformBuffers.resize(0);
}

void spotLight::destroy(VkDevice* device)
{
    destroyUniformBuffers(device, uniformBuffersHost);
    destroyUniformBuffers(device, uniformBuffersDevice);

    if(descriptorSetLayout) {vkDestroyDescriptorSetLayout(*device, descriptorSetLayout,  nullptr); descriptorSetLayout = VK_NULL_HANDLE;}
    if(shadowDescriptorSetLayout) {vkDestroyDescriptorSetLayout(*device, shadowDescriptorSetLayout,  nullptr); shadowDescriptorSetLayout = VK_NULL_HANDLE;}
    if(descriptorPool)      {vkDestroyDescriptorPool(*device, descriptorPool, nullptr); descriptorPool = VK_NULL_HANDLE;}
}

void spotLight::updateUniformBuffersFlags(std::vector<buffer>& uniformBuffers)
{
    for (auto& buffer: uniformBuffers){
        buffer.updateFlag = true;
    }
}

void spotLight::updateModelMatrix()
{
    dualQuaternion<float> dQuat = convert(rotation,translation);
    glm::mat<4,4,float,glm::defaultp> transformMatrix = convert(dQuat);
    glm::mat<4,4,float,glm::defaultp> scaleMatrix = glm::scale(glm::mat4x4(1.0f),scaling);

    modelMatrix = globalTransformation * transformMatrix * scaleMatrix;

    updateUniformBuffersFlags(uniformBuffersHost);
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
    updateUniformBuffersFlags(uniformBuffersHost);
}
void                            spotLight::setLightColor(const glm::vec4 &color){
    lightColor = color;
    updateUniformBuffersFlags(uniformBuffersHost);
}

glm::mat4x4                     spotLight::getModelMatrix() const {return modelMatrix;}
glm::vec3                       spotLight::getTranslate() const {return translation.vector();;}
glm::vec4                       spotLight::getLightColor() const {return lightColor;}
texture*                        spotLight::getTexture(){return tex;}

attachments* spotLight::getAttachments()
{
    return shadow;
}

uint8_t                         spotLight::getPipelineBitMask()     {return (enableScattering<<5)|(isShadowEnable()<<4)|(0x0);}

bool                            spotLight::isShadowEnable() const{return enableShadow;}
bool                            spotLight::isScatteringEnable() const{return enableScattering;}

VkDescriptorSet*                spotLight::getDescriptorSets(){return descriptorSets.data();}
VkDescriptorSet*                spotLight::getShadowDescriptorSets() {return shadowDescriptorSets.data();}

void spotLight::createUniformBuffers(VkPhysicalDevice* physicalDevice, VkDevice* device, uint32_t imageCount)
{
    uniformBuffersHost.resize(imageCount);
    for (auto& buffer: uniformBuffersHost){
       Buffer::create(  *physicalDevice,
                        *device,
                        sizeof(LightBufferObject),
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        &buffer.instance,
                        &buffer.memory);
       vkMapMemory(*device, buffer.memory, 0, sizeof(LightBufferObject), 0, &buffer.map);
    }
    uniformBuffersDevice.resize(imageCount);
    for (auto& buffer: uniformBuffersDevice){
       Buffer::create(  *physicalDevice,
                        *device,
                        sizeof(LightBufferObject),
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        &buffer.instance,
                        &buffer.memory);
    }
}

void spotLight::updateUniformBuffer(VkCommandBuffer commandBuffer, uint32_t frameNumber)
{
    if(uniformBuffersHost[frameNumber].updateFlag){
        LightBufferObject buffer{};
            buffer.proj = projectionMatrix;
            buffer.view = glm::inverse(modelMatrix);
            buffer.projView = projectionMatrix * buffer.view;
            buffer.position = modelMatrix * glm::vec4(0.0f,0.0f,0.0f,1.0f);
            buffer.lightColor = lightColor;
            buffer.lightProp = glm::vec4(static_cast<float>(type),lightPowerFactor,lightDropFactor,0.0f);
        std::memcpy(uniformBuffersHost[frameNumber].map, &buffer, sizeof(buffer));

        uniformBuffersHost[frameNumber].updateFlag = false;

        Buffer::copy(commandBuffer, sizeof(LightBufferObject), uniformBuffersHost[frameNumber].instance, uniformBuffersDevice[frameNumber].instance);
    }
}

void spotLight::createDescriptorPool(VkDevice* device, uint32_t imageCount)
{
    std::vector<VkDescriptorPoolSize> poolSize;
    poolSize.push_back(VkDescriptorPoolSize{});
        poolSize.back() = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(imageCount)};
    poolSize.push_back(VkDescriptorPoolSize{});
        poolSize.back() = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(imageCount)};
    poolSize.push_back(VkDescriptorPoolSize{});
        poolSize.back() = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(imageCount)};
    poolSize.push_back(VkDescriptorPoolSize{});
        poolSize.back() = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(imageCount)};
    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSize.size());
        poolInfo.pPoolSizes = poolSize.data();
        poolInfo.maxSets = static_cast<uint32_t>(2*imageCount);
    vkCreateDescriptorPool(*device, &poolInfo, nullptr, &descriptorPool);
}

namespace SpotLight {
    void createDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout){
        std::vector<VkDescriptorSetLayoutBinding> binding;
        binding.push_back(VkDescriptorSetLayoutBinding{});
            binding.back().binding = binding.size() - 1;
            binding.back().descriptorCount = 1;
            binding.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            binding.back().stageFlags = VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT;
            binding.back().pImmutableSamplers = nullptr;
        binding.push_back(VkDescriptorSetLayoutBinding{});
            binding.back().binding = binding.size() - 1;
            binding.back().descriptorCount = 1;
            binding.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            binding.back().stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            binding.back().pImmutableSamplers = nullptr;
        binding.push_back(VkDescriptorSetLayoutBinding{});
            binding.back().binding = binding.size() - 1;
            binding.back().descriptorCount = 1;
            binding.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            binding.back().stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            binding.back().pImmutableSamplers = nullptr;
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
            layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount = static_cast<uint32_t>(binding.size());
            layoutInfo.pBindings = binding.data();
        vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, descriptorSetLayout);
    }
    void createShadowDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout){
        std::vector<VkDescriptorSetLayoutBinding> binding;
        binding.push_back(VkDescriptorSetLayoutBinding{});
            binding.back().binding = binding.size() - 1;
            binding.back().descriptorCount = 1;
            binding.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            binding.back().stageFlags = VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT;
            binding.back().pImmutableSamplers = nullptr;
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
            layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount = static_cast<uint32_t>(binding.size());
            layoutInfo.pBindings = binding.data();
        vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, descriptorSetLayout);
    }
}

void spotLight::createDescriptorSets(VkDevice* device, uint32_t imageCount)
{
    SpotLight::createDescriptorSetLayout(*device,&descriptorSetLayout);
    SpotLight::createShadowDescriptorSetLayout(*device,&shadowDescriptorSetLayout);

    descriptorSets.resize(imageCount);
    std::vector<VkDescriptorSetLayout> layouts(imageCount, descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        allocInfo.pSetLayouts = layouts.data();
    vkAllocateDescriptorSets(*device, &allocInfo, descriptorSets.data());

    shadowDescriptorSets.resize(imageCount);
    std::vector<VkDescriptorSetLayout> shadowLayouts(imageCount, shadowDescriptorSetLayout);
    VkDescriptorSetAllocateInfo shadowAllocInfo{};
        shadowAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        shadowAllocInfo.descriptorPool = descriptorPool;
        shadowAllocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        shadowAllocInfo.pSetLayouts = shadowLayouts.data();
    vkAllocateDescriptorSets(*device, &shadowAllocInfo, shadowDescriptorSets.data());
}

void spotLight::updateDescriptorSets(VkDevice* device, uint32_t imageCount, texture* emptyTexture)
{
    for (size_t i=0; i<imageCount; i++)
    {
        VkDescriptorBufferInfo lightBufferInfo{};
            lightBufferInfo.buffer = uniformBuffersDevice[i].instance;
            lightBufferInfo.offset = 0;
            lightBufferInfo.range = sizeof(LightBufferObject);
        VkDescriptorImageInfo shadowImageInfo{};
            shadowImageInfo.imageLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            shadowImageInfo.imageView       = isShadowEnable() ? shadow->imageView[i] : *emptyTexture->getTextureImageView();
            shadowImageInfo.sampler         = isShadowEnable() ? shadow->sampler : *emptyTexture->getTextureSampler();
        VkDescriptorImageInfo lightTexture{};
            lightTexture.imageLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            lightTexture.imageView          = tex ? *tex->getTextureImageView() : *emptyTexture->getTextureImageView();
            lightTexture.sampler            = tex ? *tex->getTextureSampler() : *emptyTexture->getTextureSampler();

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = descriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pBufferInfo = &lightBufferInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = descriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &shadowImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = descriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &lightTexture;
        vkUpdateDescriptorSets(*device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);

        std::vector<VkWriteDescriptorSet> shadowDescriptorWrites;
        shadowDescriptorWrites.push_back(VkWriteDescriptorSet{});
            shadowDescriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            shadowDescriptorWrites.back().dstSet = shadowDescriptorSets[i];
            shadowDescriptorWrites.back().dstBinding = static_cast<uint32_t>(shadowDescriptorWrites.size() - 1);
            shadowDescriptorWrites.back().dstArrayElement = 0;
            shadowDescriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            shadowDescriptorWrites.back().descriptorCount = 1;
            shadowDescriptorWrites.back().pBufferInfo = &lightBufferInfo;
        vkUpdateDescriptorSets(*device, static_cast<uint32_t>(shadowDescriptorWrites.size()), shadowDescriptorWrites.data(), 0, nullptr);
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


