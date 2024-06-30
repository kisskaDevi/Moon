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

SpotLight::~SpotLight(){delete tex;}

SpotLight& SpotLight::updateModelMatrix() {
    moon::math::Matrix<float,4,4> transformMatrix = convert(convert(rotation, translation));
    modelMatrix = globalTransformation * transformMatrix * moon::math::scale(scaling);
    utils::raiseFlags(uniformBuffersHost);
    return *this;
}

SpotLight& SpotLight::setGlobalTransform(const moon::math::Matrix<float,4,4> & transform)
{
    globalTransformation = transform;
    return updateModelMatrix();
}

SpotLight& SpotLight::translate(const moon::math::Vector<float,3> & translate)
{
    translation += moon::math::Quaternion<float>(0.0f,translate);
    return updateModelMatrix();
}

SpotLight& SpotLight::rotate(const float & ang ,const moon::math::Vector<float,3> & ax)
{
    rotation = convert(ang, moon::math::Vector<float,3>(normalize(ax))) * rotation;
    return updateModelMatrix();
}

SpotLight& SpotLight::rotate(const moon::math::Quaternion<float>& quat)
{
    rotation = quat*rotation;
    return updateModelMatrix();
}

SpotLight& SpotLight::scale(const moon::math::Vector<float,3> & scale)
{
    scaling = scale;
    return updateModelMatrix();
}

SpotLight& SpotLight::setTranslation(const moon::math::Vector<float,3>& translate)
{
    translation = moon::math::Quaternion<float>(0.0f, translate);
    return updateModelMatrix();
}

SpotLight& SpotLight::setRotation(const float & ang ,const moon::math::Vector<float,3> & ax)
{
    rotation = convert(ang,moon::math::Vector<float,3>(normalize(ax)));
    return updateModelMatrix();
}

SpotLight& SpotLight::setRotation(const moon::math::Quaternion<float>& rotation)
{
    this->rotation = rotation;
    return updateModelMatrix();
}

SpotLight& SpotLight::rotateX(const float & ang ,const moon::math::Vector<float,3> & ax)
{
    rotationX = convert(ang,moon::math::Vector<float,3>(normalize(ax))) * rotationX;
    rotation = rotationY * rotationX;
    return updateModelMatrix();
}

SpotLight& SpotLight::rotateY(const float & ang ,const moon::math::Vector<float,3> & ax)
{
    rotationY = convert(ang, moon::math::Vector<float,3>(normalize(ax))) * rotationY;
    rotation = rotationY * rotationX;
    return updateModelMatrix();
}

void SpotLight::setTexture(moon::utils::Texture* tex) {
    this->tex = tex;
}

void SpotLight::setProjectionMatrix(const moon::math::Matrix<float,4,4> & projection)  {
    projectionMatrix = projection;
    utils::raiseFlags(uniformBuffersHost);
}

void SpotLight::setLightColor(const moon::math::Vector<float,4> &color){
    lightColor = color;
    utils::raiseFlags(uniformBuffersHost);
}

void SpotLight::setLightDropFactor(const float& dropFactor){
    lightDropFactor = dropFactor;
    utils::raiseFlags(uniformBuffersHost);
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
    if(!this->device){
        CHECK_M(device.instance == VK_NULL_HANDLE, std::string("[ deferredGraphics::bindLightSource ] VkPhysicalDevice is VK_NULL_HANDLE"));
        CHECK_M(device.getLogical() == VK_NULL_HANDLE, std::string("[ deferredGraphics::bindLightSource ] VkDevice is VK_NULL_HANDLE"));
        CHECK_M(commandPool == VK_NULL_HANDLE, std::string("[ deferredGraphics::bindLightSource ] VkCommandPool is VK_NULL_HANDLE"));

        this->device = &device;

        emptyTextureBlack = utils::Texture::empty(device, commandPool);
        emptyTextureWhite = utils::Texture::empty(device, commandPool, false);

        if(tex){
            VkCommandBuffer commandBuffer = moon::utils::singleCommandBuffer::create(device.getLogical(),commandPool);
            tex->create(device.instance, device.getLogical(), commandBuffer);
            moon::utils::singleCommandBuffer::submit(device.getLogical(),device.getQueue(0,0),commandPool,&commandBuffer);
            tex->destroyCache();
        }
        createUniformBuffers(imageCount);
        createDescriptorPool(imageCount);
        createDescriptorSets(imageCount);
        updateDescriptorSets(imageCount);
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

void SpotLight::createUniformBuffers(uint32_t imageCount)
{
    uniformBuffersHost.resize(imageCount);
    for (auto& buffer: uniformBuffersHost){
        buffer.create(device->instance, device->getLogical(), sizeof(LightBufferObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        moon::utils::Memory::instance().nameMemory(buffer, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", spotLight::createUniformBuffers, uniformBuffersHost " + std::to_string(&buffer - &uniformBuffersHost[0]));
    }
    uniformBuffersDevice.resize(imageCount);
    for (auto& buffer: uniformBuffersDevice){
        buffer.create(device->instance, device->getLogical(), sizeof(LightBufferObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        moon::utils::Memory::instance().nameMemory(buffer, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", spotLight::createUniformBuffers, uniformBuffersDevice " + std::to_string(&buffer - &uniformBuffersDevice[0]));
    }
}

void SpotLight::update(
    uint32_t frameNumber,
    VkCommandBuffer commandBuffer)
{
    if(auto& buffer = uniformBuffersHost[frameNumber]; buffer.dropFlag()){
        LightBufferObject lightBuffer{};
            lightBuffer.proj = transpose(projectionMatrix);
            lightBuffer.view = transpose(inverse(modelMatrix));
            lightBuffer.projView = transpose(projectionMatrix * inverse(modelMatrix));
            lightBuffer.position = modelMatrix * moon::math::Vector<float,4>(0.0f,0.0f,0.0f,1.0f);
            lightBuffer.lightColor = lightColor;
            lightBuffer.lightProp = moon::math::Vector<float,4>(static_cast<float>(type),lightPowerFactor,lightDropFactor,0.0f);
        buffer.copy(&lightBuffer);
        moon::utils::buffer::copy(commandBuffer, sizeof(LightBufferObject), buffer, uniformBuffersDevice[frameNumber]);
    }
}

void SpotLight::createDescriptorPool(uint32_t imageCount) {
    textureDescriptorSetLayout = moon::interfaces::Light::createTextureDescriptorSetLayout(device->getLogical());
    descriptorSetLayout = moon::interfaces::Light::createBufferDescriptorSetLayout(device->getLogical());
    CHECK(descriptorPool.create(device->getLogical(), { &textureDescriptorSetLayout , &descriptorSetLayout}, imageCount));
}

void SpotLight::createDescriptorSets(uint32_t imageCount) {
    textureDescriptorSets.resize(imageCount);
    std::vector<VkDescriptorSetLayout> textLayouts(imageCount, textureDescriptorSetLayout);
    VkDescriptorSetAllocateInfo textAllocInfo{};
        textAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        textAllocInfo.descriptorPool = descriptorPool;
        textAllocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        textAllocInfo.pSetLayouts = textLayouts.data();
    CHECK(vkAllocateDescriptorSets(device->getLogical(), &textAllocInfo, textureDescriptorSets.data()));

    descriptorSets.resize(imageCount);
    std::vector<VkDescriptorSetLayout> bufferLayouts(imageCount, descriptorSetLayout);
    VkDescriptorSetAllocateInfo bufferAllocInfo{};
        bufferAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        bufferAllocInfo.descriptorPool = descriptorPool;
        bufferAllocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        bufferAllocInfo.pSetLayouts = bufferLayouts.data();
    CHECK(vkAllocateDescriptorSets(device->getLogical(), &bufferAllocInfo, descriptorSets.data()));
}

void SpotLight::updateDescriptorSets(uint32_t imageCount)
{
    for (size_t i = 0; i < imageCount; i++)
    {
        VkDescriptorImageInfo lightTexture{};
            lightTexture.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            lightTexture.imageView = tex ? tex->imageView() : emptyTextureBlack.imageView();
            lightTexture.sampler = tex ? tex->sampler() : emptyTextureBlack.sampler();
        std::vector<VkWriteDescriptorSet> descriptorWrites;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = textureDescriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &lightTexture;
        vkUpdateDescriptorSets(device->getLogical(), static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);

        VkDescriptorBufferInfo lightBufferInfo{};
            lightBufferInfo.buffer = uniformBuffersDevice[i];
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
        vkUpdateDescriptorSets(device->getLogical(), static_cast<uint32_t>(bufferDescriptorWrites.size()), bufferDescriptorWrites.data(), 0, nullptr);
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

IsotropicLight& IsotropicLight::updateModelMatrix()
{
    moon::math::Matrix<float,4,4> transformMatrix = convert(convert(rotation, translation));
    modelMatrix = globalTransformation * transformMatrix * moon::math::scale(scaling);
    for(auto& source: lightSource)
        source->setGlobalTransform(modelMatrix);

    return *this;
}

IsotropicLight& IsotropicLight::setGlobalTransform(const moon::math::Matrix<float,4,4> & transform)
{
    globalTransformation = transform;
    return updateModelMatrix();
}

IsotropicLight& IsotropicLight::translate(const moon::math::Vector<float,3> & translate)
{
    translation += moon::math::Quaternion<float>(0.0f, translate);
    return updateModelMatrix();
}

IsotropicLight& IsotropicLight::rotate(const float & ang ,const moon::math::Vector<float,3> & ax)
{
    rotation = convert(ang, moon::math::Vector<float,3>(normalize(ax))) * rotation;
    return updateModelMatrix();
}

IsotropicLight& IsotropicLight::scale(const moon::math::Vector<float,3> & scale)
{
    scaling = scale;
    return updateModelMatrix();
}

IsotropicLight& IsotropicLight::rotateX(const float & ang ,const moon::math::Vector<float,3> & ax)
{
    rotationX = convert(ang, moon::math::Vector<float,3>(normalize(ax))) * rotationX;
    rotation = rotationY * rotationX;
    return updateModelMatrix();
}

IsotropicLight& IsotropicLight::rotateY(const float & ang ,const moon::math::Vector<float,3> & ax)
{
    rotationY = convert(ang, moon::math::Vector<float,3>(normalize(ax))) * rotationY;
    rotation = rotationY * rotationX;
    return updateModelMatrix();
}

IsotropicLight& IsotropicLight::setTranslation(const moon::math::Vector<float,3>& translate)
{
    translation = moon::math::Quaternion<float>(0.0f, translate);
    return updateModelMatrix();
}

}
