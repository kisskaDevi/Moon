#include "light.h"
#include "object.h"
#include "core/operations.h"
#include "core/graphics/deferredGraphics/renderStages/shadowGraphics.h"
#include "core/texture.h"

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
    for(size_t i=0;i<uniformBuffers.size();i++)
    {
        if (uniformBuffers[i] != VK_NULL_HANDLE)
        {
            vkDestroyBuffer(*device, uniformBuffers.at(i), nullptr);
            vkFreeMemory(*device, uniformBuffersMemory.at(i), nullptr);
            uniformBuffers[i] = VK_NULL_HANDLE;
        }
    }
    uniformBuffers.resize(0);
}

void spotLight::destroy(VkDevice* device)
{
    if (descriptorSetLayout != VK_NULL_HANDLE){
        vkDestroyDescriptorSetLayout(*device, descriptorSetLayout,  nullptr);
        descriptorSetLayout = VK_NULL_HANDLE;
    }

    if (descriptorPool != VK_NULL_HANDLE){
        vkDestroyDescriptorPool(*device, descriptorPool, nullptr);
        descriptorPool = VK_NULL_HANDLE;
    }

    if(shadow){
        shadow->destroy();
    }
}

void spotLight::setGlobalTransform(const glm::mat4 & transform)
{
    m_globalTransform = transform;
    updateModelMatrix();
}

void spotLight::translate(const glm::vec3 & translate)
{
    m_translate += translate;
    updateModelMatrix();
}

void spotLight::rotate(const float & ang ,const glm::vec3 & ax)
{
    m_rotate = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax))*m_rotate;
    updateModelMatrix();
}

void spotLight::scale(const glm::vec3 & scale)
{
    m_scale = scale;
    updateModelMatrix();
}

void spotLight::setPosition(const glm::vec3& translate)
{
    m_translate = translate;
    updateModelMatrix();
}

void spotLight::rotateX(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    m_rotateX = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax)) * m_rotateX;
    m_rotate = m_rotateX * m_rotateY;
    updateModelMatrix();
}

void spotLight::rotateY(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    m_rotateY = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax)) * m_rotateY;
    m_rotate = m_rotateX * m_rotateY;
    updateModelMatrix();
}

void spotLight::updateModelMatrix()
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

void                            spotLight::setLightColor(const glm::vec4 &color)                {lightColor = color;}
void                            spotLight::setShadowExtent(const VkExtent2D & shadowExtent)     {this->shadowExtent = shadowExtent;}
void                            spotLight::setShadow(bool enable)                               {enableShadow = enable;}
void                            spotLight::setScattering(bool enable)                           {enableScattering = enable;}
void                            spotLight::setTexture(texture* tex)                             {this->tex = tex;}
void                            spotLight::setProjectionMatrix(const glm::mat4x4 & projection)  {projectionMatrix = projection;}

glm::mat4x4                     spotLight::getModelMatrix() const {return modelMatrix;}
glm::vec3                       spotLight::getTranslate() const {return m_translate;}
glm::vec4                       spotLight::getLightColor() const {return lightColor;}
texture*                        spotLight::getTexture(){return tex;}

uint8_t                         spotLight::getPipelineBitMask()     {return (enableScattering<<5)|(isShadowEnable()<<4)|(0x0);}

bool                            spotLight::isShadowEnable() const{return shadow ? true : false;}
bool                            spotLight::isScatteringEnable() const{return enableScattering;}

VkDescriptorSet*                spotLight::getDescriptorSets(){return descriptorSets.data();}
VkCommandBuffer*                spotLight::getShadowCommandBuffer() {return shadow->getCommandBuffer().data();}

void spotLight::createUniformBuffers(VkPhysicalDevice* physicalDevice, VkDevice* device, uint32_t imageCount)
{
    uniformBuffers.resize(imageCount);
    uniformBuffersMemory.resize(imageCount);
    for (size_t i = 0; i < imageCount; i++){
        createBuffer(   physicalDevice,
                        device,
                        sizeof(LightBufferObject),
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        uniformBuffers[i],
                        uniformBuffersMemory[i]);
    }
}

void spotLight::updateUniformBuffer(VkDevice* device, uint32_t frameNumber)
{
    LightBufferObject buffer{};
        buffer.proj = projectionMatrix;
        buffer.view = glm::inverse(modelMatrix);
        buffer.projView = projectionMatrix * buffer.view;
        buffer.position = modelMatrix * glm::vec4(0.0f,0.0f,0.0f,1.0f);
        buffer.lightColor = lightColor;
        buffer.lightProp = glm::vec4(static_cast<float>(type),lightPowerFactor,lightDropFactor,0.0f);
    void* data;
    vkMapMemory(*device, uniformBuffersMemory[frameNumber], 0, sizeof(buffer), 0, &data);
        memcpy(data, &buffer, sizeof(buffer));
    vkUnmapMemory(*device, uniformBuffersMemory[frameNumber]);
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
    if(shadow){
        shadow->updateDescriptorSets(uniformBuffers.size(),uniformBuffers.data(),sizeof(LightBufferObject));
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
    if (vkCreateDescriptorPool(*device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create light descriptor pool!");
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
    if (vkAllocateDescriptorSets(*device, &allocInfo, descriptorSets.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate SpotLightingPass descriptor sets!");
}

void spotLight::updateDescriptorSets(VkDevice* device, uint32_t imageCount, texture* emptyTexture)
{
    for (size_t i=0; i<imageCount; i++)
    {
        uint32_t index = 0;

        VkDescriptorBufferInfo lightBufferInfo{};
            lightBufferInfo.buffer = uniformBuffers[i];
            lightBufferInfo.offset = 0;
            lightBufferInfo.range = sizeof(LightBufferObject);
        VkDescriptorImageInfo shadowImageInfo{};
            shadowImageInfo.imageLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            shadowImageInfo.imageView       = isShadowEnable() ? shadow->getImageView() : emptyTexture->getTextureImageView();
            shadowImageInfo.sampler         = isShadowEnable() ? shadow->getSampler() : emptyTexture->getTextureSampler();
        VkDescriptorImageInfo lightTexture{};
            lightTexture.imageLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            lightTexture.imageView          = tex ? tex->getTextureImageView() : emptyTexture->getTextureImageView();
            lightTexture.sampler            = tex ? tex->getTextureSampler() : emptyTexture->getTextureSampler();

        index = 0;
        std::array<VkWriteDescriptorSet,3> descriptorWrites{};
            descriptorWrites.at(index).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.at(index).dstSet = descriptorSets[i];
            descriptorWrites.at(index).dstBinding = index;
            descriptorWrites.at(index).dstArrayElement = 0;
            descriptorWrites.at(index).descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.at(index).descriptorCount = 1;
            descriptorWrites.at(index).pBufferInfo = &lightBufferInfo;
        index++;
            descriptorWrites.at(index).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.at(index).dstSet = descriptorSets[i];
            descriptorWrites.at(index).dstBinding = index;
            descriptorWrites.at(index).dstArrayElement = 0;
            descriptorWrites.at(index).descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.at(index).descriptorCount = 1;
            descriptorWrites.at(index).pImageInfo = &shadowImageInfo;
        index++;
            descriptorWrites.at(index).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.at(index).dstSet = descriptorSets[i];
            descriptorWrites.at(index).dstBinding = index;
            descriptorWrites.at(index).dstArrayElement = 0;
            descriptorWrites.at(index).descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.at(index).descriptorCount = 1;
            descriptorWrites.at(index).pImageInfo = &lightTexture;
        vkUpdateDescriptorSets(*device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

//======================================================================================================================//
//============//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//============//
//============//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//============//
//============//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//============//
//======================================================================================================================//

isotropicLight::isotropicLight(std::vector<spotLight *>& lightSource)
{
    m_scale = glm::vec3(1.0f,1.0f,1.0f);
    m_globalTransform = glm::mat4x4(1.0f);
    m_translate = glm::vec3(0.0f,0.0f,0.0f);
    m_rotate = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_rotateX = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_rotateY = glm::quat(1.0f,0.0f,0.0f,0.0f);

    uint32_t number = lightSource.size();
    uint32_t index = number;
    lightSource.push_back(new spotLight(true,false,spotType::square));
    lightSource.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    lightSource.at(index)->setLightColor(glm::vec4(1.0f,0.0f,0.0f,1.0f));

    index++;
    lightSource.push_back(new spotLight(true,false,spotType::square));
    lightSource.at(index)->rotate(glm::radians(-90.0f),glm::vec3(1.0f,0.0f,0.0f));
    lightSource.at(index)->setLightColor(glm::vec4(0.0f,1.0f,0.0f,1.0f));

    index++;
    lightSource.push_back(new spotLight(true,false,spotType::square));
    lightSource.at(index)->setLightColor(glm::vec4(0.0f,0.0f,1.0f,1.0f));

    index++;
    lightSource.push_back(new spotLight(true,false,spotType::square));
    lightSource.at(index)->rotate(glm::radians(90.0f),glm::vec3(0.0f,1.0f,0.0f));
    lightSource.at(index)->setLightColor(glm::vec4(0.3f,0.6f,0.9f,1.0f));

    index++;
    lightSource.push_back(new spotLight(true,false,spotType::square));
    lightSource.at(index)->rotate(glm::radians(-90.0f),glm::vec3(0.0f,1.0f,0.0f));
    lightSource.at(index)->setLightColor(glm::vec4(0.6f,0.9f,0.3f,1.0f));

    index++;
    lightSource.push_back(new spotLight(true,false,spotType::square));
    lightSource.at(index)->rotate(glm::radians(180.0f),glm::vec3(1.0f,0.0f,0.0f));
    lightSource.at(index)->setLightColor(glm::vec4(0.9f,0.3f,0.6f,1.0f));

    this->lightSource.resize(6);
    for(uint32_t i=0;i<6;i++){
        this->lightSource[i] = lightSource[number+i];
    }
}

isotropicLight::~isotropicLight(){}

glm::vec4       isotropicLight::getLightColor() const {return lightColor;}
glm::vec3       isotropicLight::getTranslate() const {return m_translate;}

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

void isotropicLight::setGlobalTransform(const glm::mat4 & transform)
{
    m_globalTransform = transform;
    updateViewMatrix();
}

void isotropicLight::translate(const glm::vec3 & translate)
{
    m_translate += translate;
    updateViewMatrix();
}

void isotropicLight::rotate(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    m_rotate = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax))*m_rotate;
    updateViewMatrix();
}

void isotropicLight::scale(const glm::vec3 & scale)
{
    m_scale = scale;
    updateViewMatrix();
}

void isotropicLight::setPosition(const glm::vec3& translate)
{
    m_translate = translate;
    updateViewMatrix();
}

void isotropicLight::updateViewMatrix()
{
    glm::mat4x4 translateMatrix = glm::translate(glm::mat4x4(1.0f),-m_translate);
    glm::mat4x4 rotateMatrix = glm::mat4x4(1.0f);
    if(!(m_rotate.x==0&&m_rotate.y==0&&m_rotate.z==0))
        rotateMatrix = glm::rotate(glm::mat4x4(1.0f),2.0f*glm::acos(m_rotate.w),glm::vec3(m_rotate.x,m_rotate.y,m_rotate.z));
    glm::mat4x4 scaleMatrix = glm::scale(glm::mat4x4(1.0f),m_scale);
    glm::mat4x4 localMatrix = m_globalTransform * translateMatrix * rotateMatrix * scaleMatrix;

    for(uint32_t i=0;i<6;i++)
        lightSource.at(i)->setGlobalTransform(localMatrix);
}

void isotropicLight::rotateX(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    m_rotateX = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax)) * m_rotateX;
    m_rotate = m_rotateX * m_rotateY;
    updateViewMatrix();
}

void isotropicLight::rotateY(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    m_rotateY = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax)) * m_rotateY;
    updateViewMatrix();
}



