#include "light.h"
#include "object.h"
#include "core/operations.h"
#include "core/graphics/shadowGraphics.h"
#include "gltfmodel.h"

spotLight::spotLight(uint32_t type): type(type)
{
    m_scale = glm::vec3(1.0f,1.0f,1.0f);
    m_globalTransform = glm::mat4x4(1.0f);
    m_translate = glm::vec3(0.0f,0.0f,0.0f);
    m_rotate = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_rotateX = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_rotateY = glm::quat(1.0f,0.0f,0.0f,0.0f);
    viewMatrix = glm::mat4x4(1.0f);
    modelMatrix = glm::mat4x4(1.0f);
    lightColor = glm::vec4(0.0f);
    lightPowerFactor = 1.0f;
    lightDropFactor = 0.1f;
}

spotLight::spotLight(const std::string & TEXTURE_PATH, uint32_t type): type(type)
{
    m_scale = glm::vec3(1.0f,1.0f,1.0f);
    m_globalTransform = glm::mat4x4(1.0f);
    m_translate = glm::vec3(0.0f,0.0f,0.0f);
    m_rotate = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_rotateX = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_rotateY = glm::quat(1.0f,0.0f,0.0f,0.0f);
    viewMatrix = glm::mat4x4(1.0f);
    modelMatrix = glm::mat4x4(1.0f);
    lightColor = glm::vec4(0.0f);
    lightPowerFactor = 1.0f;
    lightDropFactor = 0.1f;

    tex = new texture(TEXTURE_PATH);
}

spotLight::~spotLight(){
    delete tex;
}

void spotLight::destroyBuffer(VkDevice* device)
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

void spotLight::cleanup(VkDevice* device)
{
    if (descriptorPool != VK_NULL_HANDLE){
        vkDestroyDescriptorPool(*device, descriptorPool, nullptr);
        descriptorPool = VK_NULL_HANDLE;
    }

    if(enableShadow){
        shadow->destroy();
        enableShadow = false;
    }
}

void spotLight::setGlobalTransform(const glm::mat4 & transform)
{
    m_globalTransform = transform;
    updateViewMatrix();
}

void spotLight::translate(const glm::vec3 & translate)
{
    m_translate += translate;
    updateViewMatrix();
}

void spotLight::rotate(const float & ang ,const glm::vec3 & ax)
{
    m_rotate = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax))*m_rotate;
    updateViewMatrix();
}

void spotLight::scale(const glm::vec3 & scale)
{
    m_scale = scale;
    updateViewMatrix();
}

void spotLight::setPosition(const glm::vec3& translate)
{
    m_translate = translate;
    updateViewMatrix();
}

void spotLight::updateViewMatrix()
{
    glm::mat4x4 translateMatrix = glm::translate(glm::mat4x4(1.0f),m_translate);
    glm::mat4x4 rotateMatrix = glm::mat4x4(1.0f);
    if(!(m_rotate.x==0&&m_rotate.y==0&&m_rotate.z==0))
    {
        rotateMatrix = glm::rotate(glm::mat4x4(1.0f),2.0f*glm::acos(m_rotate.w),glm::vec3(m_rotate.x,m_rotate.y,m_rotate.z));
    }
    glm::mat4x4 scaleMatrix = glm::scale(glm::mat4x4(1.0f),m_scale);
    modelMatrix = m_globalTransform * translateMatrix * rotateMatrix * scaleMatrix;
    viewMatrix = glm::inverse(modelMatrix);
}

void spotLight::rotateX(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    m_rotateX = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax)) * m_rotateX;
    m_rotate = m_rotateX * m_rotateY;
    updateViewMatrix();
}

void spotLight::rotateY(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    m_rotateY = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax)) * m_rotateY;
    m_rotate = m_rotateX * m_rotateY;
    updateViewMatrix();
}

void spotLight::setProjectionMatrix(const glm::mat4x4 & projection)
{
    projectionMatrix = projection;
}

void spotLight::createShadow(VkPhysicalDevice* physicalDevice, VkDevice* device, QueueFamilyIndices* queueFamilyIndices, uint32_t imageCount)
{
    enableShadow = true;
    shadow = new shadowGraphics(imageCount,shadowExtent);
    shadow->setDeviceProp(physicalDevice,device,queueFamilyIndices);
    shadow->createShadow();
}

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

void spotLight::updateLightBuffer(VkDevice* device, uint32_t frameNumber)
{
    LightBufferObject buffer{};
        buffer.proj = projectionMatrix;
        buffer.view = viewMatrix;
        buffer.projView = projectionMatrix * viewMatrix;
        buffer.position = modelMatrix * glm::vec4(0.0f,0.0f,0.0f,1.0f);
        buffer.lightColor = lightColor;
        buffer.lightProp = glm::vec4(static_cast<float>(type),lightPowerFactor,lightDropFactor,0.0f);
    void* data;
    vkMapMemory(*device, uniformBuffersMemory[frameNumber], 0, sizeof(buffer), 0, &data);
        memcpy(data, &buffer, sizeof(buffer));
    vkUnmapMemory(*device, uniformBuffersMemory[frameNumber]);
}

void                            spotLight::setLightColor(const glm::vec4 &color){this->lightColor = color;}
void                            spotLight::setShadowExtent(const VkExtent2D & shadowExtent){this->shadowExtent=shadowExtent;}
void                            spotLight::setScattering(bool enable){this->enableScattering=enable;}
void                            spotLight::setTexture(texture* tex){this->tex=tex;}


VkDescriptorPool&               spotLight::getDescriptorPool(){return descriptorPool;}
std::vector<VkDescriptorSet>&   spotLight::getDescriptorSets(){return descriptorSets;}
std::vector<VkBuffer>&          spotLight::getUniformBuffers(){return uniformBuffers;}

glm::mat4x4                     spotLight::getViewMatrix() const {return viewMatrix;}
glm::mat4x4                     spotLight::getModelMatrix() const {return modelMatrix;}
glm::vec3                       spotLight::getTranslate() const {return m_translate;}

glm::vec4                       spotLight::getLightColor() const {return lightColor;}
bool                            spotLight::isShadowEnable() const{return enableShadow;}
bool                            spotLight::isScatteringEnable() const{return enableScattering;}

texture                         *spotLight::getTexture(){return tex;}


void                            spotLight::updateShadowCommandBuffer(uint32_t frameNumber, std::vector<object*>& objects){
    shadow->updateCommandBuffer(frameNumber,objects);
}
void                            spotLight::createShadowCommandBuffers(){
    shadow->createCommandBuffers();
}
void                            spotLight::updateShadowDescriptorSets(){
    shadow->updateDescriptorSets(uniformBuffers.size(),uniformBuffers.data(),sizeof(LightBufferObject));
}
std::vector<VkCommandBuffer>&   spotLight::getShadowCommandBuffer(){
    return shadow->getCommandBuffer();
}
VkImageView&                    spotLight::getShadowImageView(){
    return shadow->getImageView();
}
VkSampler&                      spotLight::getShadowSampler(){
    return shadow->getSampler();
}

//======================================================================================================================//
//============//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//============//
//============//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//============//
//============//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//============//
//======================================================================================================================//

pointLight::pointLight(std::vector<spotLight *>& lightSource)
{
    m_scale = glm::vec3(1.0f,1.0f,1.0f);
    m_globalTransform = glm::mat4x4(1.0f);
    m_translate = glm::vec3(0.0f,0.0f,0.0f);
    m_rotate = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_rotateX = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_rotateY = glm::quat(1.0f,0.0f,0.0f,0.0f);

    uint32_t number = lightSource.size();
    uint32_t index = number;
    lightSource.push_back(new spotLight(spotType::square));
    lightSource.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    lightSource.at(index)->setLightColor(glm::vec4(1.0f,0.0f,0.0f,1.0f));

    index++;
    lightSource.push_back(new spotLight(spotType::square));
    lightSource.at(index)->rotate(glm::radians(-90.0f),glm::vec3(1.0f,0.0f,0.0f));
    lightSource.at(index)->setLightColor(glm::vec4(0.0f,1.0f,0.0f,1.0f));

    index++;
    lightSource.push_back(new spotLight(spotType::square));
    lightSource.at(index)->setLightColor(glm::vec4(0.0f,0.0f,1.0f,1.0f));

    index++;
    lightSource.push_back(new spotLight(spotType::square));
    lightSource.at(index)->rotate(glm::radians(90.0f),glm::vec3(0.0f,1.0f,0.0f));
    lightSource.at(index)->setLightColor(glm::vec4(0.3f,0.6f,0.9f,1.0f));

    index++;
    lightSource.push_back(new spotLight(spotType::square));
    lightSource.at(index)->rotate(glm::radians(-90.0f),glm::vec3(0.0f,1.0f,0.0f));
    lightSource.at(index)->setLightColor(glm::vec4(0.6f,0.9f,0.3f,1.0f));

    index++;
    lightSource.push_back(new spotLight(spotType::square));
    lightSource.at(index)->rotate(glm::radians(180.0f),glm::vec3(1.0f,0.0f,0.0f));
    lightSource.at(index)->setLightColor(glm::vec4(0.9f,0.3f,0.6f,1.0f));

    this->lightSource.resize(6);
    for(uint32_t i=0;i<6;i++){
        this->lightSource[i] = lightSource[number+i];
    }
}

pointLight::~pointLight(){}

glm::vec4       pointLight::getLightColor() const {return lightColor;}
glm::vec3       pointLight::getTranslate() const {return m_translate;}

void pointLight::setProjectionMatrix(const glm::mat4x4 & projection)
{
    projectionMatrix = projection;
    for(uint32_t i=0;i<6;i++)
        lightSource.at(i)->setProjectionMatrix(projectionMatrix);
}

void pointLight::setLightColor(const glm::vec4 &color)
{
    this->lightColor = color;
    for(uint32_t i=0;i<6;i++)
        lightSource.at(i)->setLightColor(color);
}

void pointLight::setGlobalTransform(const glm::mat4 & transform)
{
    m_globalTransform = transform;
    updateViewMatrix();
}

void pointLight::translate(const glm::vec3 & translate)
{
    m_translate += translate;
    updateViewMatrix();
}

void pointLight::rotate(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    m_rotate = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax))*m_rotate;
    updateViewMatrix();
}

void pointLight::scale(const glm::vec3 & scale)
{
    m_scale = scale;
    updateViewMatrix();
}

void pointLight::setPosition(const glm::vec3& translate)
{
    m_translate = translate;
    updateViewMatrix();
}

void pointLight::updateViewMatrix()
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

void pointLight::rotateX(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    m_rotateX = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax)) * m_rotateX;
    m_rotate = m_rotateX * m_rotateY;
    updateViewMatrix();
}

void pointLight::rotateY(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    m_rotateY = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax)) * m_rotateY;
    updateViewMatrix();
}

