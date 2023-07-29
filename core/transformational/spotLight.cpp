#include "spotLight.h"
#include "attachments.h"
#include "operations.h"
#include "texture.h"
#include "dualQuaternion.h"

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

spotLight::spotLight(const std::filesystem::path & TEXTURE_PATH, bool enableShadow, bool enableScattering, uint32_t type):
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

void spotLight::destroyUniformBuffers(VkDevice device, std::vector<buffer>& uniformBuffers)
{
    for(auto& buffer: uniformBuffers){
        buffer.destroy(device);
    }
    uniformBuffers.clear();
}

void spotLight::destroy(VkDevice device)
{
    destroyUniformBuffers(device, uniformBuffersHost);
    destroyUniformBuffers(device, uniformBuffersDevice);

    if(descriptorSetLayout) {vkDestroyDescriptorSetLayout(device, descriptorSetLayout,  nullptr); descriptorSetLayout = VK_NULL_HANDLE;}
    if(bufferDescriptorSetLayout) {vkDestroyDescriptorSetLayout(device, bufferDescriptorSetLayout,  nullptr); bufferDescriptorSetLayout = VK_NULL_HANDLE;}
    if(descriptorPool)      {vkDestroyDescriptorPool(device, descriptorPool, nullptr); descriptorPool = VK_NULL_HANDLE;}
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
    matrix<float,4,4> transformMatrix = convert(dQuat);

    modelMatrix = globalTransformation * transformMatrix * ::scale(scaling);

    updateUniformBuffersFlags(uniformBuffersHost);
}

void spotLight::setGlobalTransform(const matrix<float,4,4> & transform)
{
    globalTransformation = transform;
    updateModelMatrix();
}

void spotLight::translate(const vector<float,3> & translate)
{
    translation += quaternion<float>(0.0f,translate);
    updateModelMatrix();
}

void spotLight::rotate(const float & ang ,const vector<float,3> & ax)
{
    rotation = convert(ang,vector<float,3>(normalize(ax)))*rotation;
    updateModelMatrix();
}

void spotLight::scale(const vector<float,3> & scale)
{
    scaling = scale;
    updateModelMatrix();
}

void spotLight::setPosition(const vector<float,3>& translate)
{
    translation = quaternion<float>(0.0f,vector<float,3>(translate[0],translate[1],translate[2]));
    updateModelMatrix();
}

void spotLight::rotateX(const float & ang ,const vector<float,3> & ax)
{
    rotationX = convert(ang,vector<float,3>(normalize(ax))) * rotationX;
    rotation = rotationY * rotationX;
    updateModelMatrix();
}

void spotLight::rotateY(const float & ang ,const vector<float,3> & ax)
{
    rotationY = convert(ang,vector<float,3>(normalize(ax))) * rotationY;
    rotation = rotationY * rotationX;
    updateModelMatrix();
}

void                            spotLight::setShadowExtent(const VkExtent2D & shadowExtent)     {this->shadowExtent = shadowExtent;}
void                            spotLight::setShadow(bool enable)                               {enableShadow = enable;}
void                            spotLight::setScattering(bool enable)                           {enableScattering = enable;}
void                            spotLight::setTexture(texture* tex)                             {this->tex = tex;}
void                            spotLight::setProjectionMatrix(const matrix<float,4,4> & projection)  {
    projectionMatrix = projection;
    updateUniformBuffersFlags(uniformBuffersHost);
}
void                            spotLight::setLightColor(const vector<float,4> &color){
    lightColor = color;
    updateUniformBuffersFlags(uniformBuffersHost);
}

matrix<float,4,4>               spotLight::getModelMatrix() const {return modelMatrix;}
vector<float,3>                 spotLight::getTranslate() const {return vector<float,3>(translation.vector()[0],translation.vector()[1],translation.vector()[2]);}
vector<float,4>                 spotLight::getLightColor() const {return lightColor;}
texture*                        spotLight::getTexture(){return tex;}

attachments* spotLight::getAttachments()
{
    return shadow;
}

uint8_t                         spotLight::getPipelineBitMask()     {return (enableScattering<<5)|(isShadowEnable()<<4)|(0x0);}

bool                            spotLight::isShadowEnable() const{return enableShadow;}
bool                            spotLight::isScatteringEnable() const{return enableScattering;}

VkDescriptorSet*                spotLight::getDescriptorSets(){return descriptorSets.data();}
VkDescriptorSet*                spotLight::getBufferDescriptorSets() {return bufferDescriptorSets.data();}

void spotLight::createUniformBuffers(VkPhysicalDevice physicalDevice, VkDevice device, uint32_t imageCount)
{
    uniformBuffersHost.resize(imageCount);
    for (auto& buffer: uniformBuffersHost){
       Buffer::create(  physicalDevice,
                        device,
                        sizeof(LightBufferObject),
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        &buffer.instance,
                        &buffer.memory);
       vkMapMemory(device, buffer.memory, 0, sizeof(LightBufferObject), 0, &buffer.map);
    }
    uniformBuffersDevice.resize(imageCount);
    for (auto& buffer: uniformBuffersDevice){
       Buffer::create(  physicalDevice,
                        device,
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
            buffer.proj = transpose(projectionMatrix);
            buffer.view = transpose(inverse(modelMatrix));
            buffer.projView = transpose(projectionMatrix * inverse(modelMatrix));
            buffer.position = modelMatrix * vector<float,4>(0.0f,0.0f,0.0f,1.0f);
            buffer.lightColor = lightColor;
            buffer.lightProp = vector<float,4>(static_cast<float>(type),lightPowerFactor,lightDropFactor,0.0f);
        std::memcpy(uniformBuffersHost[frameNumber].map, &buffer, sizeof(buffer));

        uniformBuffersHost[frameNumber].updateFlag = false;

        Buffer::copy(commandBuffer, sizeof(LightBufferObject), uniformBuffersHost[frameNumber].instance, uniformBuffersDevice[frameNumber].instance);
    }
}

void spotLight::createDescriptorPool(VkDevice device, uint32_t imageCount)
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
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);
}

void spotLight::createDescriptorSets(VkDevice device, uint32_t imageCount)
{
    light::createTextureDescriptorSetLayout(device,&descriptorSetLayout);
    light::createBufferDescriptorSetLayout(device,&bufferDescriptorSetLayout);

    descriptorSets.resize(imageCount);
    std::vector<VkDescriptorSetLayout> layouts(imageCount, descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        allocInfo.pSetLayouts = layouts.data();
    vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data());

    bufferDescriptorSets.resize(imageCount);
    std::vector<VkDescriptorSetLayout> shadowLayouts(imageCount, bufferDescriptorSetLayout);
    VkDescriptorSetAllocateInfo shadowAllocInfo{};
        shadowAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        shadowAllocInfo.descriptorPool = descriptorPool;
        shadowAllocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        shadowAllocInfo.pSetLayouts = shadowLayouts.data();
    vkAllocateDescriptorSets(device, &shadowAllocInfo, bufferDescriptorSets.data());
}

void spotLight::updateDescriptorSets(VkDevice device, uint32_t imageCount, texture* emptyTexture)
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
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);

        std::vector<VkWriteDescriptorSet> bufferDescriptorWrites;
        bufferDescriptorWrites.push_back(VkWriteDescriptorSet{});
            bufferDescriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            bufferDescriptorWrites.back().dstSet = bufferDescriptorSets[i];
            bufferDescriptorWrites.back().dstBinding = static_cast<uint32_t>(bufferDescriptorWrites.size() - 1);
            bufferDescriptorWrites.back().dstArrayElement = 0;
            bufferDescriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            bufferDescriptorWrites.back().descriptorCount = 1;
            bufferDescriptorWrites.back().pBufferInfo = &lightBufferInfo;
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(bufferDescriptorWrites.size()), bufferDescriptorWrites.data(), 0, nullptr);
    }
}

//isotropicLight

isotropicLight::isotropicLight(std::vector<spotLight *>& lightSource)
{
    lightSource.push_back(new spotLight(true,false,spotType::square));
    lightSource.back()->rotate(radians(90.0f),vector<float,3>(1.0f,0.0f,0.0f));
    lightSource.back()->setLightColor(vector<float,4>(1.0f,0.0f,0.0f,1.0f));
    this->lightSource.push_back(lightSource.back());

    lightSource.push_back(new spotLight(true,false,spotType::square));
    lightSource.back()->rotate(radians(-90.0f),vector<float,3>(1.0f,0.0f,0.0f));
    lightSource.back()->setLightColor(vector<float,4>(0.0f,1.0f,0.0f,1.0f));
    this->lightSource.push_back(lightSource.back());

    lightSource.push_back(new spotLight(true,false,spotType::square));
    lightSource.back()->rotate(radians(0.0f),vector<float,3>(0.0f,1.0f,0.0f));
    lightSource.back()->setLightColor(vector<float,4>(0.0f,0.0f,1.0f,1.0f));
    this->lightSource.push_back(lightSource.back());

    lightSource.push_back(new spotLight(true,false,spotType::square));
    lightSource.back()->rotate(radians(90.0f),vector<float,3>(0.0f,1.0f,0.0f));
    lightSource.back()->setLightColor(vector<float,4>(0.3f,0.6f,0.9f,1.0f));
    this->lightSource.push_back(lightSource.back());

    lightSource.push_back(new spotLight(true,false,spotType::square));
    lightSource.back()->rotate(radians(-90.0f),vector<float,3>(0.0f,1.0f,0.0f));
    lightSource.back()->setLightColor(vector<float,4>(0.6f,0.9f,0.3f,1.0f));
    this->lightSource.push_back(lightSource.back());

    lightSource.push_back(new spotLight(true,false,spotType::square));
    lightSource.back()->rotate(radians(180.0f),vector<float,3>(1.0f,0.0f,0.0f));
    lightSource.back()->setLightColor(vector<float,4>(0.9f,0.3f,0.6f,1.0f));
    this->lightSource.push_back(lightSource.back());
}

isotropicLight::~isotropicLight(){}

vector<float,4>       isotropicLight::getLightColor() const {return lightColor;}
vector<float,3>       isotropicLight::getTranslate() const {return vector<float,3>(translation.vector()[0],translation.vector()[1],translation.vector()[2]);}

void isotropicLight::setProjectionMatrix(const matrix<float,4,4> & projection)
{
    projectionMatrix = projection;
    for(auto& source: lightSource)
        source->setProjectionMatrix(projectionMatrix);
}

void isotropicLight::setLightColor(const vector<float,4> &color)
{
    this->lightColor = color;
    for(auto& source: lightSource)
        source->setLightColor(color);
}

void isotropicLight::updateModelMatrix()
{
    dualQuaternion<float> dQuat = convert(rotation,translation);
    matrix<float,4,4> transformMatrix = convert(dQuat);

    modelMatrix = globalTransformation * transformMatrix * ::scale(scaling);

    for(auto& source: lightSource)
        source->setGlobalTransform(modelMatrix);
}

void isotropicLight::setGlobalTransform(const matrix<float,4,4> & transform)
{
    globalTransformation = transform;
    updateModelMatrix();
}

void isotropicLight::translate(const vector<float,3> & translate)
{
    translation += quaternion<float>(0.0f, vector<float,3>(translate[0],translate[1],translate[2]));
    updateModelMatrix();
}

void isotropicLight::rotate(const float & ang ,const vector<float,3> & ax)
{
    rotation = convert(ang,vector<float,3>(normalize(ax)))*rotation;
    updateModelMatrix();
}

void isotropicLight::scale(const vector<float,3> & scale)
{
    scaling = scale;
    updateModelMatrix();
}

void isotropicLight::rotateX(const float & ang ,const vector<float,3> & ax)
{
    rotationX = convert(ang,vector<float,3>(normalize(ax))) * rotationX;
    rotation = rotationY * rotationX;
    updateModelMatrix();
}

void isotropicLight::rotateY(const float & ang ,const vector<float,3> & ax)
{
    rotationY = convert(ang,vector<float,3>(normalize(ax))) * rotationY;
    rotation = rotationY * rotationX;
    updateModelMatrix();
}

void isotropicLight::setPosition(const vector<float,3>& translate)
{
    translation = quaternion<float>(0.0f,vector<float,3>(translate[0],translate[1],translate[2]));
    updateModelMatrix();
}


