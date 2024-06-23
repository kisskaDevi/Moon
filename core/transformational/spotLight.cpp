#include "spotLight.h"
#include "operations.h"
#include "dualQuaternion.h"
#include "device.h"

#include <cstring>

namespace moon::transformational {

SpotLight::SpotLight(const moon::math::Vector<float,4>& color, const moon::math::Matrix<float,4,4> & projection, bool enableShadow, bool enableScattering, SpotType type):
    lightColor(color),
    projectionMatrix(projection),
    type(type)
{
    pipelineBitMask = moon::interfaces::LightType::spot;
    this->enableShadow = enableShadow;
    this->enableScattering = enableScattering;
}

SpotLight::SpotLight(const std::filesystem::path & TEXTURE_PATH, const moon::math::Matrix<float,4,4> & projection, bool enableShadow, bool enableScattering, SpotType type):
    tex(new moon::utils::Texture(TEXTURE_PATH)),
    projectionMatrix(projection),
    type(type)
{
    pipelineBitMask = moon::interfaces::LightType::spot;
    this->enableShadow = enableShadow;
    this->enableScattering = enableScattering;
}

SpotLight::~SpotLight(){
    SpotLight::destroy(device);
    delete tex;
}

void SpotLight::destroy(VkDevice device)
{
    destroyBuffers(device, uniformBuffersHost);
    destroyBuffers(device, uniformBuffersDevice);

    if(descriptorPool) {vkDestroyDescriptorPool(device, descriptorPool, nullptr); descriptorPool = VK_NULL_HANDLE;}

    emptyTextureBlack.destroy(device);
    emptyTextureWhite.destroy(device);

    if(tex){
        tex->destroy(device);
    }

    created = false;
}

void SpotLight::updateUniformBuffersFlags(std::vector<moon::utils::Buffer>& uniformBuffers)
{
    for (auto& buffer: uniformBuffers){
        buffer.updateFlag = true;
    }
}

void SpotLight::updateModelMatrix()
{
    moon::math::DualQuaternion<float> dQuat = convert(rotation,translation);
    moon::math::Matrix<float,4,4> transformMatrix = convert(dQuat);

    modelMatrix = globalTransformation * transformMatrix * moon::math::scale(scaling);

    updateUniformBuffersFlags(uniformBuffersHost);
}

SpotLight& SpotLight::setGlobalTransform(const moon::math::Matrix<float,4,4> & transform)
{
    globalTransformation = transform;
    updateModelMatrix();
    return *this;
}

SpotLight& SpotLight::translate(const moon::math::Vector<float,3> & translate)
{
    translation += moon::math::Quaternion<float>(0.0f,translate);
    updateModelMatrix();
    return *this;
}

SpotLight& SpotLight::rotate(const float & ang ,const moon::math::Vector<float,3> & ax)
{
    rotation = convert(ang, moon::math::Vector<float,3>(normalize(ax))) * rotation;
    updateModelMatrix();
    return *this;
}

SpotLight& SpotLight::rotate(const moon::math::Quaternion<float>& quat)
{
    rotation = quat*rotation;
    updateModelMatrix();
    return *this;
}

SpotLight& SpotLight::scale(const moon::math::Vector<float,3> & scale)
{
    scaling = scale;
    updateModelMatrix();
    return *this;
}

SpotLight& SpotLight::setTranslation(const moon::math::Vector<float,3>& translate)
{
    translation = moon::math::Quaternion<float>(0.0f,moon::math::Vector<float,3>(translate[0],translate[1],translate[2]));
    updateModelMatrix();
    return *this;
}

SpotLight& SpotLight::setRotation(const float & ang ,const moon::math::Vector<float,3> & ax)
{
    rotation = convert(ang,moon::math::Vector<float,3>(normalize(ax)));
    updateModelMatrix();
    return *this;
}

SpotLight& SpotLight::setRotation(const moon::math::Quaternion<float>& rotation)
{
    this->rotation = rotation;
    updateModelMatrix();
    return *this;
}

SpotLight& SpotLight::rotateX(const float & ang ,const moon::math::Vector<float,3> & ax)
{
    rotationX = convert(ang,moon::math::Vector<float,3>(normalize(ax))) * rotationX;
    rotation = rotationY * rotationX;
    updateModelMatrix();
    return *this;
}

SpotLight& SpotLight::rotateY(const float & ang ,const moon::math::Vector<float,3> & ax)
{
    rotationY = convert(ang, moon::math::Vector<float,3>(normalize(ax))) * rotationY;
    rotation = rotationY * rotationX;
    updateModelMatrix();
    return *this;
}

void SpotLight::setTexture(moon::utils::Texture* tex) {
    this->tex = tex;
}

void SpotLight::setProjectionMatrix(const moon::math::Matrix<float,4,4> & projection)  {
    projectionMatrix = projection;
    updateUniformBuffersFlags(uniformBuffersHost);
}

void SpotLight::setLightColor(const moon::math::Vector<float,4> &color){
    lightColor = color;
    updateUniformBuffersFlags(uniformBuffersHost);
}

void SpotLight::setLightDropFactor(const float& dropFactor){
    lightDropFactor = dropFactor;
    updateUniformBuffersFlags(uniformBuffersHost);
}

moon::math::Matrix<float,4,4> SpotLight::getModelMatrix() const {
    return modelMatrix;
}

moon::math::Vector<float,3> SpotLight::getTranslate() const {
    return translation.im();
}

moon::math::Vector<float,4> SpotLight::getLightColor() const {
    return lightColor;
}

void SpotLight::create(
    const moon::utils::PhysicalDevice& device,
    VkCommandPool commandPool,
    uint32_t imageCount)
{
    if(!created){
        CHECK_M(device.instance == VK_NULL_HANDLE, std::string("[ deferredGraphics::bindLightSource ] VkPhysicalDevice is VK_NULL_HANDLE"));
        CHECK_M(device.getLogical() == VK_NULL_HANDLE, std::string("[ deferredGraphics::bindLightSource ] VkDevice is VK_NULL_HANDLE"));
        CHECK_M(commandPool == VK_NULL_HANDLE, std::string("[ deferredGraphics::bindLightSource ] VkCommandPool is VK_NULL_HANDLE"));

        emptyTextureBlack = createEmptyTexture(device, commandPool);
        emptyTextureWhite = createEmptyTexture(device, commandPool, false);

        if(tex){
            VkCommandBuffer commandBuffer = moon::utils::singleCommandBuffer::create(device.getLogical(),commandPool);
            tex->createTextureImage(device.instance, device.getLogical(), commandBuffer);
            moon::utils::singleCommandBuffer::submit(device.getLogical(),device.getQueue(0,0),commandPool,&commandBuffer);
            tex->destroyStagingBuffer(device.getLogical());
            tex->createTextureImageView(device.getLogical());
            tex->createTextureSampler(device.getLogical(),{VK_FILTER_LINEAR,VK_FILTER_LINEAR,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT,VK_SAMPLER_ADDRESS_MODE_REPEAT});
        }
        createUniformBuffers(device.instance,device.getLogical(),imageCount);
        createDescriptorPool(device.getLogical(), imageCount);
        createDescriptorSets(device.getLogical(), imageCount);
        updateDescriptorSets(device.getLogical(), imageCount);
        created = true;
        this->device = device.getLogical();
    }
}

void SpotLight::render(
    uint32_t frameNumber,
    VkCommandBuffer commandBuffer,
    const std::vector<VkDescriptorSet>& descriptorSet,
    VkPipelineLayout pipelineLayout,
    VkPipeline pipeline)
{
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    std::vector<VkDescriptorSet> descriptorSets = descriptorSet;
    descriptorSets.push_back(this->descriptorSets[frameNumber]);
    descriptorSets.push_back(textureDescriptorSets[frameNumber]);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, static_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(), 0, nullptr);
    vkCmdDraw(commandBuffer, 18, 1, 0, 0);
}

void SpotLight::createUniformBuffers(VkPhysicalDevice physicalDevice, VkDevice device, uint32_t imageCount)
{
    uniformBuffersHost.resize(imageCount);
    for (auto& buffer: uniformBuffersHost){
        moon::utils::buffer::create(  physicalDevice,
                        device,
                        sizeof(LightBufferObject),
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        &buffer.instance,
                        &buffer.memory);
        CHECK(vkMapMemory(device, buffer.memory, 0, sizeof(LightBufferObject), 0, &buffer.map));

        moon::utils::Memory::instance().nameMemory(buffer.memory, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", spotLight::createUniformBuffers, uniformBuffersHost " + std::to_string(&buffer - &uniformBuffersHost[0]));
    }
    uniformBuffersDevice.resize(imageCount);
    for (auto& buffer: uniformBuffersDevice){
        moon::utils::buffer::create(  physicalDevice,
                        device,
                        sizeof(LightBufferObject),
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        &buffer.instance,
                        &buffer.memory);

        moon::utils::Memory::instance().nameMemory(buffer.memory, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", spotLight::createUniformBuffers, uniformBuffersDevice " + std::to_string(&buffer - &uniformBuffersDevice[0]));
    }
}

void SpotLight::update(
    uint32_t frameNumber,
    VkCommandBuffer commandBuffer)
{
    if(uniformBuffersHost[frameNumber].updateFlag){
        LightBufferObject buffer{};
            buffer.proj = transpose(projectionMatrix);
            buffer.view = transpose(inverse(modelMatrix));
            buffer.projView = transpose(projectionMatrix * inverse(modelMatrix));
            buffer.position = modelMatrix * moon::math::Vector<float,4>(0.0f,0.0f,0.0f,1.0f);
            buffer.lightColor = lightColor;
            buffer.lightProp = moon::math::Vector<float,4>(static_cast<float>(type),lightPowerFactor,lightDropFactor,0.0f);
        std::memcpy(uniformBuffersHost[frameNumber].map, &buffer, sizeof(buffer));

        uniformBuffersHost[frameNumber].updateFlag = false;

        moon::utils::buffer::copy(commandBuffer, sizeof(LightBufferObject), uniformBuffersHost[frameNumber].instance, uniformBuffersDevice[frameNumber].instance);
    }
}

void SpotLight::createDescriptorPool(VkDevice device, uint32_t imageCount)
{
    std::vector<VkDescriptorPoolSize> poolSize = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(imageCount)},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(imageCount)}
    };
    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSize.size());
        poolInfo.pPoolSizes = poolSize.data();
        poolInfo.maxSets = static_cast<uint32_t>(2 * imageCount);
    CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool));
}

void SpotLight::createDescriptorSets(VkDevice device, uint32_t imageCount)
{
    textureDescriptorSetLayout = moon::interfaces::Light::createTextureDescriptorSetLayout(device);
    descriptorSetLayout = moon::interfaces::Light::createBufferDescriptorSetLayout(device);

    textureDescriptorSets.resize(imageCount);
    std::vector<VkDescriptorSetLayout> textLayouts(imageCount, textureDescriptorSetLayout);
    VkDescriptorSetAllocateInfo textAllocInfo{};
        textAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        textAllocInfo.descriptorPool = descriptorPool;
        textAllocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        textAllocInfo.pSetLayouts = textLayouts.data();
    CHECK(vkAllocateDescriptorSets(device, &textAllocInfo, textureDescriptorSets.data()));

    descriptorSets.resize(imageCount);
    std::vector<VkDescriptorSetLayout> bufferLayouts(imageCount, descriptorSetLayout);
    VkDescriptorSetAllocateInfo bufferAllocInfo{};
        bufferAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        bufferAllocInfo.descriptorPool = descriptorPool;
        bufferAllocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        bufferAllocInfo.pSetLayouts = bufferLayouts.data();
    CHECK(vkAllocateDescriptorSets(device, &bufferAllocInfo, descriptorSets.data()));
}

void SpotLight::updateDescriptorSets(VkDevice device, uint32_t imageCount)
{
    for (size_t i = 0; i < imageCount; i++)
    {
        VkDescriptorImageInfo lightTexture{};
            lightTexture.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            lightTexture.imageView = tex ? *tex->getTextureImageView() : *emptyTextureBlack.getTextureImageView();
            lightTexture.sampler = tex ? *tex->getTextureSampler() : *emptyTextureBlack.getTextureSampler();
        std::vector<VkWriteDescriptorSet> descriptorWrites;
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
            bufferDescriptorWrites.back().dstSet = descriptorSets[i];
            bufferDescriptorWrites.back().dstBinding = static_cast<uint32_t>(bufferDescriptorWrites.size() - 1);
            bufferDescriptorWrites.back().dstArrayElement = 0;
            bufferDescriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            bufferDescriptorWrites.back().descriptorCount = 1;
            bufferDescriptorWrites.back().pBufferInfo = &lightBufferInfo;
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(bufferDescriptorWrites.size()), bufferDescriptorWrites.data(), 0, nullptr);
    }
}

void SpotLight::printStatus() const
{
    std::cout << "translation\t" << translation.im()[0] << '\t' << translation.im()[1] << '\t' << translation.im()[2] << '\n';
    std::cout << "rotation\t" << rotation.re() << '\t' << rotation.im()[0] << '\t' << rotation.im()[1] << '\t' << rotation.im()[2] << '\n';
    std::cout << "scale\t" << scaling[0] << '\t' << scaling[1] << '\t' << scaling[2] << '\n';
}

//isotropicLight

IsotropicLight::IsotropicLight(const moon::math::Vector<float,4>& color, float radius)
{
    const auto proj = moon::math::perspective(moon::math::radians(91.0f), 1.0f, 0.01f, radius);

    lightSource.push_back(new SpotLight(color, proj,true,false,SpotType::square));
    lightSource.back()->rotate(moon::math::radians(90.0f), moon::math::Vector<float,3>(1.0f,0.0f,0.0f));

    lightSource.push_back(new SpotLight(color, proj,true,false,SpotType::square));
    lightSource.back()->rotate(moon::math::radians(-90.0f), moon::math::Vector<float,3>(1.0f,0.0f,0.0f));

    lightSource.push_back(new SpotLight(color, proj,true,false,SpotType::square));
    lightSource.back()->rotate(moon::math::radians(0.0f), moon::math::Vector<float,3>(0.0f,1.0f,0.0f));

    lightSource.push_back(new SpotLight(color, proj,true,false,SpotType::square));
    lightSource.back()->rotate(moon::math::radians(90.0f), moon::math::Vector<float,3>(0.0f,1.0f,0.0f));

    lightSource.push_back(new SpotLight(color, proj,true,false,SpotType::square));
    lightSource.back()->rotate(moon::math::radians(-90.0f), moon::math::Vector<float,3>(0.0f,1.0f,0.0f));

    lightSource.push_back(new SpotLight(color, proj,true,false,SpotType::square));
    lightSource.back()->rotate(moon::math::radians(180.0f), moon::math::Vector<float,3>(1.0f,0.0f,0.0f));

    // colors for debug if color = {0, 0, 0, 0}
    if(dot(color,color) == 0.0f){
        lightSource.at(0)->setLightColor(moon::math::Vector<float,4>(1.0f,0.0f,0.0f,1.0f));
        lightSource.at(1)->setLightColor(moon::math::Vector<float,4>(0.0f,1.0f,0.0f,1.0f));
        lightSource.at(2)->setLightColor(moon::math::Vector<float,4>(0.0f,0.0f,1.0f,1.0f));
        lightSource.at(3)->setLightColor(moon::math::Vector<float,4>(0.3f,0.6f,0.9f,1.0f));
        lightSource.at(4)->setLightColor(moon::math::Vector<float,4>(0.6f,0.9f,0.3f,1.0f));
        lightSource.at(5)->setLightColor(moon::math::Vector<float,4>(0.9f,0.3f,0.6f,1.0f));
    }
}

IsotropicLight::~IsotropicLight(){}

moon::math::Vector<float,4> IsotropicLight::getLightColor() const {
    return lightColor;
}

moon::math::Vector<float,3> IsotropicLight::getTranslate() const {
    return translation.im();
}

std::vector<SpotLight*> IsotropicLight::get() const {
    return lightSource;
}

void IsotropicLight:: setProjectionMatrix(const moon::math::Matrix<float,4,4> & projection)
{
    projectionMatrix = projection;
    for(auto& source: lightSource)
        source->setProjectionMatrix(projectionMatrix);
}

void IsotropicLight::setLightColor(const moon::math::Vector<float,4> &color)
{
    this->lightColor = color;
    for(auto& source: lightSource)
        source->setLightColor(color);
}

void IsotropicLight::setLightDropFactor(const float& dropFactor){
    lightDropFactor = dropFactor;
    for(auto& source: lightSource)
        source->setLightDropFactor(lightDropFactor);
}

void IsotropicLight::updateModelMatrix()
{
    moon::math::DualQuaternion<float> dQuat = convert(rotation,translation);
    moon::math::Matrix<float,4,4> transformMatrix = convert(dQuat);

    modelMatrix = globalTransformation * transformMatrix * moon::math::scale(scaling);

    for(auto& source: lightSource)
        source->setGlobalTransform(modelMatrix);
}

IsotropicLight& IsotropicLight::setGlobalTransform(const moon::math::Matrix<float,4,4> & transform)
{
    globalTransformation = transform;
    updateModelMatrix();
    return *this;
}

IsotropicLight& IsotropicLight::translate(const moon::math::Vector<float,3> & translate)
{
    translation += moon::math::Quaternion<float>(0.0f, moon::math::Vector<float,3>(translate[0],translate[1],translate[2]));
    updateModelMatrix();
    return *this;
}

IsotropicLight& IsotropicLight::rotate(const float & ang ,const moon::math::Vector<float,3> & ax)
{
    rotation = convert(ang, moon::math::Vector<float,3>(normalize(ax))) * rotation;
    updateModelMatrix();
    return *this;
}

IsotropicLight& IsotropicLight::scale(const moon::math::Vector<float,3> & scale)
{
    scaling = scale;
    updateModelMatrix();
    return *this;
}

void IsotropicLight::rotateX(const float & ang ,const moon::math::Vector<float,3> & ax)
{
    rotationX = convert(ang, moon::math::Vector<float,3>(normalize(ax))) * rotationX;
    rotation = rotationY * rotationX;
    updateModelMatrix();
}

void IsotropicLight::rotateY(const float & ang ,const moon::math::Vector<float,3> & ax)
{
    rotationY = convert(ang, moon::math::Vector<float,3>(normalize(ax))) * rotationY;
    rotation = rotationY * rotationX;
    updateModelMatrix();
}

void IsotropicLight::setTranslation(const moon::math::Vector<float,3>& translate)
{
    translation = moon::math::Quaternion<float>(0.0f, moon::math::Vector<float,3>(translate[0],translate[1],translate[2]));
    updateModelMatrix();
}

}
