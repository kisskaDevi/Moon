#include "spotLight.h"
#include "attachments.h"
#include "operations.h"
#include "texture.h"
#include "dualQuaternion.h"
#include "device.h"

#include <cstring>

spotLight::spotLight(const matrix<float,4,4> & projection, bool enableShadow, bool enableScattering, uint32_t type):
    projectionMatrix(projection),
    type(type)
{
    this->enableShadow = enableShadow;
    this->enableScattering = enableScattering;
}

spotLight::spotLight(const std::filesystem::path & TEXTURE_PATH, const matrix<float,4,4> & projection, bool enableShadow, bool enableScattering, uint32_t type):
    tex(new texture(TEXTURE_PATH)),
    projectionMatrix(projection),
    type(type)
{
    this->enableShadow = enableShadow;
    this->enableScattering = enableScattering;
}

spotLight::~spotLight(){
    delete tex;
}

void spotLight::destroy(VkDevice device)
{
    destroyBuffers(device, uniformBuffersHost);
    destroyBuffers(device, uniformBuffersDevice);

    if(textureDescriptorSetLayout) {vkDestroyDescriptorSetLayout(device, textureDescriptorSetLayout,  nullptr); textureDescriptorSetLayout = VK_NULL_HANDLE;}
    if(bufferDescriptorSetLayout) {vkDestroyDescriptorSetLayout(device, bufferDescriptorSetLayout,  nullptr); bufferDescriptorSetLayout = VK_NULL_HANDLE;}

    if(descriptorPool) {vkDestroyDescriptorPool(device, descriptorPool, nullptr); descriptorPool = VK_NULL_HANDLE;}

    if(emptyTextureBlack){
        emptyTextureBlack->destroy(device);
        delete emptyTextureBlack;
        emptyTextureBlack = nullptr;
    }
    if(emptyTextureWhite){
        emptyTextureWhite->destroy(device);
        delete emptyTextureWhite;
        emptyTextureWhite = nullptr;
    }

    if(tex){
        tex->destroy(device);
    }

    created = false;
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

spotLight& spotLight::setGlobalTransform(const matrix<float,4,4> & transform)
{
    globalTransformation = transform;
    updateModelMatrix();
    return *this;
}

spotLight& spotLight::translate(const vector<float,3> & translate)
{
    translation += quaternion<float>(0.0f,translate);
    updateModelMatrix();
    return *this;
}

spotLight& spotLight::rotate(const float & ang ,const vector<float,3> & ax)
{
    rotation = convert(ang,vector<float,3>(normalize(ax)))*rotation;
    updateModelMatrix();
    return *this;
}

spotLight& spotLight::rotate(const quaternion<float>& quat)
{
    rotation = quat*rotation;
    updateModelMatrix();
    return *this;
}

spotLight& spotLight::scale(const vector<float,3> & scale)
{
    scaling = scale;
    updateModelMatrix();
    return *this;
}

spotLight& spotLight::setTranslation(const vector<float,3>& translate)
{
    translation = quaternion<float>(0.0f,vector<float,3>(translate[0],translate[1],translate[2]));
    updateModelMatrix();
    return *this;
}

spotLight& spotLight::setRotation(const float & ang ,const vector<float,3> & ax)
{
    rotation = convert(ang,vector<float,3>(normalize(ax)));
    updateModelMatrix();
    return *this;
}

spotLight& spotLight::setRotation(const quaternion<float>& rotation)
{
    this->rotation = rotation;
    updateModelMatrix();
    return *this;
}

spotLight& spotLight::rotateX(const float & ang ,const vector<float,3> & ax)
{
    rotationX = convert(ang,vector<float,3>(normalize(ax))) * rotationX;
    rotation = rotationY * rotationX;
    updateModelMatrix();
    return *this;
}

spotLight& spotLight::rotateY(const float & ang ,const vector<float,3> & ax)
{
    rotationY = convert(ang,vector<float,3>(normalize(ax))) * rotationY;
    rotation = rotationY * rotationX;
    updateModelMatrix();
    return *this;
}

void spotLight::setTexture(texture* tex) {this->tex = tex;}
void spotLight::setProjectionMatrix(const matrix<float,4,4> & projection)  {
    projectionMatrix = projection;
    updateUniformBuffersFlags(uniformBuffersHost);
}
void spotLight::setLightColor(const vector<float,4> &color){
    lightColor = color;
    updateUniformBuffersFlags(uniformBuffersHost);
}
void spotLight::setLightDropFactor(const float& dropFactor){
    lightDropFactor = dropFactor;
    updateUniformBuffersFlags(uniformBuffersHost);
}

matrix<float,4,4>   spotLight::getModelMatrix() const {return modelMatrix;}
vector<float,3>     spotLight::getTranslate() const {return vector<float,3>(translation.vector()[0],translation.vector()[1],translation.vector()[2]);}
vector<float,4>     spotLight::getLightColor() const {return lightColor;}

const std::vector<VkDescriptorSet>& spotLight::getDescriptorSets() const {
    return bufferDescriptorSets;
}

uint8_t spotLight::getPipelineBitMask() const {
    return 0x0;
}

void spotLight::create(
    physicalDevice device,
    VkCommandPool commandPool,
    uint32_t imageCount)
{
    if(!created){
        CHECKERROR(device.instance == VK_NULL_HANDLE, std::string("[ deferredGraphics::bindLightSource ] VkPhysicalDevice is VK_NULL_HANDLE"));
        CHECKERROR(device.getLogical() == VK_NULL_HANDLE, std::string("[ deferredGraphics::bindLightSource ] VkDevice is VK_NULL_HANDLE"));
        CHECKERROR(commandPool == VK_NULL_HANDLE, std::string("[ deferredGraphics::bindLightSource ] VkCommandPool is VK_NULL_HANDLE"));

        emptyTextureBlack = createEmptyTexture(device, commandPool);
        emptyTextureWhite = createEmptyTexture(device, commandPool, false);

        if(tex){
            VkCommandBuffer commandBuffer = SingleCommandBuffer::create(device.getLogical(),commandPool);
            tex->createTextureImage(device.instance, device.getLogical(), commandBuffer);
            SingleCommandBuffer::submit(device.getLogical(),device.getQueue(0,0),commandPool,&commandBuffer);
            tex->destroyStagingBuffer(device.getLogical());
            tex->createTextureImageView(device.getLogical());
            tex->createTextureSampler(device.getLogical(),{VK_FILTER_LINEAR,VK_FILTER_LINEAR,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT});
        }
        createUniformBuffers(device.instance,device.getLogical(),imageCount);
        createDescriptorPool(device.getLogical(), imageCount);
        createDescriptorSets(device.getLogical(), imageCount);
        updateDescriptorSets(device.getLogical(), imageCount, emptyTextureBlack, emptyTextureWhite);
        created = true;
    }
}

void spotLight::render(
    uint32_t frameNumber,
    VkCommandBuffer commandBuffer,
    VkDescriptorSet descriptorSet,
    std::unordered_map<uint8_t, VkPipelineLayout> PipelineLayoutDictionary,
    std::unordered_map<uint8_t, VkPipeline> PipelinesDictionary)
{
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, PipelinesDictionary[getPipelineBitMask()]);
    std::vector<VkDescriptorSet> descriptorSets = {descriptorSet, bufferDescriptorSets[frameNumber], textureDescriptorSets[frameNumber]};
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineLayoutDictionary[getPipelineBitMask()], 0, static_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(), 0, nullptr);
    vkCmdDraw(commandBuffer, 18, 1, 0, 0);
}

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

       Memory::instance().nameMemory(buffer.memory, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", spotLight::createUniformBuffers, uniformBuffersHost " + std::to_string(&buffer - &uniformBuffersHost[0]));
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

       Memory::instance().nameMemory(buffer.memory, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", spotLight::createUniformBuffers, uniformBuffersDevice " + std::to_string(&buffer - &uniformBuffersDevice[0]));
    }
}

void spotLight::update(
    uint32_t frameNumber,
    VkCommandBuffer commandBuffer)
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
    light::createTextureDescriptorSetLayout(device,&textureDescriptorSetLayout);
    light::createBufferDescriptorSetLayout(device,&bufferDescriptorSetLayout);

    textureDescriptorSets.resize(imageCount);
    std::vector<VkDescriptorSetLayout> layouts(imageCount, textureDescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        allocInfo.pSetLayouts = layouts.data();
    vkAllocateDescriptorSets(device, &allocInfo, textureDescriptorSets.data());

    bufferDescriptorSets.resize(imageCount);
    std::vector<VkDescriptorSetLayout> shadowLayouts(imageCount, bufferDescriptorSetLayout);
    VkDescriptorSetAllocateInfo shadowAllocInfo{};
        shadowAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        shadowAllocInfo.descriptorPool = descriptorPool;
        shadowAllocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        shadowAllocInfo.pSetLayouts = shadowLayouts.data();
    vkAllocateDescriptorSets(device, &shadowAllocInfo, bufferDescriptorSets.data());
}

void spotLight::updateDescriptorSets(VkDevice device, uint32_t imageCount, texture* emptyTextureBlack , texture* emptyTextureWhite)
{
    const auto& shadow = shadowMaps.front();
    for (size_t i=0; i<imageCount; i++)
    {
        VkDescriptorImageInfo shadowImageInfo{};
            shadowImageInfo.imageLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            shadowImageInfo.imageView       = isShadowEnable() && shadow->instances.size() > 0 ? shadow->instances[i].imageView : *emptyTextureWhite->getTextureImageView();
            shadowImageInfo.sampler         = isShadowEnable() && shadow->sampler ? shadow->sampler : *emptyTextureWhite->getTextureSampler();
        VkDescriptorImageInfo lightTexture{};
            lightTexture.imageLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            lightTexture.imageView          = tex ? *tex->getTextureImageView() : *emptyTextureBlack->getTextureImageView();
            lightTexture.sampler            = tex ? *tex->getTextureSampler() : *emptyTextureBlack->getTextureSampler();

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = textureDescriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &shadowImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = textureDescriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &lightTexture;
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);

        VkDescriptorBufferInfo lightBufferInfo{};
            lightBufferInfo.buffer = uniformBuffersDevice[i].instance;
            lightBufferInfo.offset = 0;
            lightBufferInfo.range = sizeof(LightBufferObject);
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

void spotLight::printStatus() const
{
    std::cout << "translation\t" << translation.vector()[0] << '\t' << translation.vector()[1] << '\t' << translation.vector()[2] << '\n';
    std::cout << "rotation\t" << rotation.scalar() << '\t' << rotation.vector()[0] << '\t' << rotation.vector()[1] << '\t' << rotation.vector()[2] << '\n';
    std::cout << "scale\t" << scaling[0] << '\t' << scaling[1] << '\t' << scaling[2] << '\n';
}

//isotropicLight

isotropicLight::isotropicLight(std::vector<spotLight *>& lightSource, float radius)
{
    const auto proj = perspective(radians(91.0f), 1.0f, 0.01f, radius);

    lightSource.push_back(new spotLight(proj,true,false,spotType::square));
    lightSource.back()->rotate(radians(90.0f),vector<float,3>(1.0f,0.0f,0.0f));
    lightSource.back()->setLightColor(vector<float,4>(1.0f,0.0f,0.0f,1.0f));
    this->lightSource.push_back(lightSource.back());

    lightSource.push_back(new spotLight(proj,true,false,spotType::square));
    lightSource.back()->rotate(radians(-90.0f),vector<float,3>(1.0f,0.0f,0.0f));
    lightSource.back()->setLightColor(vector<float,4>(0.0f,1.0f,0.0f,1.0f));
    this->lightSource.push_back(lightSource.back());

    lightSource.push_back(new spotLight(proj,true,false,spotType::square));
    lightSource.back()->rotate(radians(0.0f),vector<float,3>(0.0f,1.0f,0.0f));
    lightSource.back()->setLightColor(vector<float,4>(0.0f,0.0f,1.0f,1.0f));
    this->lightSource.push_back(lightSource.back());

    lightSource.push_back(new spotLight(proj,true,false,spotType::square));
    lightSource.back()->rotate(radians(90.0f),vector<float,3>(0.0f,1.0f,0.0f));
    lightSource.back()->setLightColor(vector<float,4>(0.3f,0.6f,0.9f,1.0f));
    this->lightSource.push_back(lightSource.back());

    lightSource.push_back(new spotLight(proj,true,false,spotType::square));
    lightSource.back()->rotate(radians(-90.0f),vector<float,3>(0.0f,1.0f,0.0f));
    lightSource.back()->setLightColor(vector<float,4>(0.6f,0.9f,0.3f,1.0f));
    this->lightSource.push_back(lightSource.back());

    lightSource.push_back(new spotLight(proj,true,false,spotType::square));
    lightSource.back()->rotate(radians(180.0f),vector<float,3>(1.0f,0.0f,0.0f));
    lightSource.back()->setLightColor(vector<float,4>(0.9f,0.3f,0.6f,1.0f));
    this->lightSource.push_back(lightSource.back());
}

isotropicLight::~isotropicLight(){}

vector<float,4>       isotropicLight::getLightColor() const {return lightColor;}
vector<float,3>       isotropicLight::getTranslate() const {return vector<float,3>(translation.vector()[0],translation.vector()[1],translation.vector()[2]);}

void isotropicLight:: setProjectionMatrix(const matrix<float,4,4> & projection)
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

void isotropicLight::setLightDropFactor(const float& dropFactor){
    lightDropFactor = dropFactor;
    for(auto& source: lightSource)
        source->setLightDropFactor(lightDropFactor);
}

void isotropicLight::updateModelMatrix()
{
    dualQuaternion<float> dQuat = convert(rotation,translation);
    matrix<float,4,4> transformMatrix = convert(dQuat);

    modelMatrix = globalTransformation * transformMatrix * ::scale(scaling);

    for(auto& source: lightSource)
        source->setGlobalTransform(modelMatrix);
}

isotropicLight& isotropicLight::setGlobalTransform(const matrix<float,4,4> & transform)
{
    globalTransformation = transform;
    updateModelMatrix();
    return *this;
}

isotropicLight& isotropicLight::translate(const vector<float,3> & translate)
{
    translation += quaternion<float>(0.0f, vector<float,3>(translate[0],translate[1],translate[2]));
    updateModelMatrix();
    return *this;
}

isotropicLight& isotropicLight::rotate(const float & ang ,const vector<float,3> & ax)
{
    rotation = convert(ang,vector<float,3>(normalize(ax)))*rotation;
    updateModelMatrix();
    return *this;
}

isotropicLight& isotropicLight::scale(const vector<float,3> & scale)
{
    scaling = scale;
    updateModelMatrix();
    return *this;
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

void isotropicLight::setTranslation(const vector<float,3>& translate)
{
    translation = quaternion<float>(0.0f,vector<float,3>(translate[0],translate[1],translate[2]));
    updateModelMatrix();
}


