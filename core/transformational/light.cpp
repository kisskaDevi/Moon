#include "light.h"
#include "object.h"
#include "core/operations.h"
#include "core/graphics/shadowGraphics.h"
#include "gltfmodel.h"

light<spotLight>::light(uint32_t type): type(type)
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

light<spotLight>::light(const std::string & TEXTURE_PATH, uint32_t type): type(type)
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

light<spotLight>::~light(){
    delete tex;
}

void light<spotLight>::destroyBuffer(VkDevice* device)
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

void light<spotLight>::cleanup(VkDevice* device)
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

void light<spotLight>::setGlobalTransform(const glm::mat4 & transform)
{
    m_globalTransform = transform;
    updateViewMatrix();
}

void light<spotLight>::translate(const glm::vec3 & translate)
{
    m_translate += translate;
    updateViewMatrix();
}

void light<spotLight>::rotate(const float & ang ,const glm::vec3 & ax)
{
    m_rotate = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax))*m_rotate;
    updateViewMatrix();
}

void light<spotLight>::scale(const glm::vec3 & scale)
{
    m_scale = scale;
    updateViewMatrix();
}

void light<spotLight>::setPosition(const glm::vec3& translate)
{
    m_translate = translate;
    updateViewMatrix();
}

void light<spotLight>::updateViewMatrix()
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

void light<spotLight>::rotateX(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    m_rotateX = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax)) * m_rotateX;
    m_rotate = m_rotateX * m_rotateY;
    updateViewMatrix();
}

void light<spotLight>::rotateY(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    m_rotateY = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax)) * m_rotateY;
    m_rotate = m_rotateX * m_rotateY;
    updateViewMatrix();
}

void light<spotLight>::setProjectionMatrix(const glm::mat4x4 & projection)
{
    projectionMatrix = projection;
}

void light<spotLight>::createShadow(VkPhysicalDevice* physicalDevice, VkDevice* device, QueueFamilyIndices* queueFamilyIndices, uint32_t imageCount)
{
    enableShadow = true;
    shadow = new shadowGraphics(imageCount,shadowExtent);
    shadow->setDeviceProp(physicalDevice,device,queueFamilyIndices);
    shadow->createShadow();
}

void light<spotLight>::createUniformBuffers(VkPhysicalDevice* physicalDevice, VkDevice* device, uint32_t imageCount)
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

void light<spotLight>::updateLightBuffer(VkDevice* device, uint32_t frameNumber)
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

void                            light<spotLight>::setLightColor(const glm::vec4 &color){this->lightColor = color;}
void                            light<spotLight>::setShadowExtent(const VkExtent2D & shadowExtent){this->shadowExtent=shadowExtent;}
void                            light<spotLight>::setScattering(bool enable){this->enableScattering=enable;}
void                            light<spotLight>::setTexture(texture* tex){this->tex=tex;}


VkDescriptorPool&               light<spotLight>::getDescriptorPool(){return descriptorPool;}
std::vector<VkDescriptorSet>&   light<spotLight>::getDescriptorSets(){return descriptorSets;}
std::vector<VkBuffer>&          light<spotLight>::getUniformBuffers(){return uniformBuffers;}

glm::mat4x4                     light<spotLight>::getViewMatrix() const {return viewMatrix;}
glm::mat4x4                     light<spotLight>::getModelMatrix() const {return modelMatrix;}
glm::vec3                       light<spotLight>::getTranslate() const {return m_translate;}

glm::vec4                       light<spotLight>::getLightColor() const {return lightColor;}
bool                            light<spotLight>::isShadowEnable() const{return enableShadow;}
bool                            light<spotLight>::isScatteringEnable() const{return enableScattering;}

texture                         *light<spotLight>::getTexture(){return tex;}


void                            light<spotLight>::updateShadowCommandBuffer(uint32_t frameNumber, ShadowPassObjects objects){
    shadow->updateCommandBuffer(frameNumber,objects);
}
void                            light<spotLight>::createShadowCommandBuffers(){
    shadow->createCommandBuffers();
}
void                            light<spotLight>::updateShadowDescriptorSets(){
    shadow->updateDescriptorSets(uniformBuffers.size(),uniformBuffers.data(),sizeof(LightBufferObject));
}
std::vector<VkCommandBuffer>&   light<spotLight>::getShadowCommandBuffer(){
    return shadow->getCommandBuffer();
}
VkImageView&                    light<spotLight>::getShadowImageView(){
    return shadow->getImageView();
}
VkSampler&                      light<spotLight>::getShadowSampler(){
    return shadow->getSampler();
}

//======================================================================================================================//
//============//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//============//
//============//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//============//
//============//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//============//
//======================================================================================================================//

light<pointLight>::light(std::vector<light<spotLight> *>& lightSource)
{
    m_scale = glm::vec3(1.0f,1.0f,1.0f);
    m_globalTransform = glm::mat4x4(1.0f);
    m_translate = glm::vec3(0.0f,0.0f,0.0f);
    m_rotate = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_rotateX = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_rotateY = glm::quat(1.0f,0.0f,0.0f,0.0f);

    uint32_t number = lightSource.size();
    uint32_t index = number;
    lightSource.push_back(new light<spotLight>(lightType::point));
    lightSource.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    lightSource.at(index)->setLightColor(glm::vec4(1.0f,0.0f,0.0f,1.0f));

    index++;
    lightSource.push_back(new light<spotLight>(lightType::point));
    lightSource.at(index)->rotate(glm::radians(-90.0f),glm::vec3(1.0f,0.0f,0.0f));
    lightSource.at(index)->setLightColor(glm::vec4(0.0f,1.0f,0.0f,1.0f));

    index++;
    lightSource.push_back(new light<spotLight>(lightType::point));
    lightSource.at(index)->setLightColor(glm::vec4(0.0f,0.0f,1.0f,1.0f));

    index++;
    lightSource.push_back(new light<spotLight>(lightType::point));
    lightSource.at(index)->rotate(glm::radians(90.0f),glm::vec3(0.0f,1.0f,0.0f));
    lightSource.at(index)->setLightColor(glm::vec4(0.3f,0.6f,0.9f,1.0f));

    index++;
    lightSource.push_back(new light<spotLight>(lightType::point));
    lightSource.at(index)->rotate(glm::radians(-90.0f),glm::vec3(0.0f,1.0f,0.0f));
    lightSource.at(index)->setLightColor(glm::vec4(0.6f,0.9f,0.3f,1.0f));

    index++;
    lightSource.push_back(new light<spotLight>(lightType::point));
    lightSource.at(index)->rotate(glm::radians(180.0f),glm::vec3(1.0f,0.0f,0.0f));
    lightSource.at(index)->setLightColor(glm::vec4(0.9f,0.3f,0.6f,1.0f));

    this->lightSource.resize(6);
    for(uint32_t i=0;i<6;i++){
        this->lightSource[i] = lightSource[number+i];
    }
}

light<pointLight>::~light(){}

glm::vec4       light<pointLight>::getLightColor() const {return lightColor;}
glm::vec3       light<pointLight>::getTranslate() const {return m_translate;}

void light<pointLight>::setProjectionMatrix(const glm::mat4x4 & projection)
{
    projectionMatrix = projection;
    for(uint32_t i=0;i<6;i++)
        lightSource.at(i)->setProjectionMatrix(projectionMatrix);
}

void light<pointLight>::setLightColor(const glm::vec4 &color)
{
    this->lightColor = color;
    for(uint32_t i=0;i<6;i++)
        lightSource.at(i)->setLightColor(color);
}

void light<pointLight>::setGlobalTransform(const glm::mat4 & transform)
{
    m_globalTransform = transform;
    updateViewMatrix();
}

void light<pointLight>::translate(const glm::vec3 & translate)
{
    m_translate += translate;
    updateViewMatrix();
}

void light<pointLight>::rotate(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    m_rotate = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax))*m_rotate;
    updateViewMatrix();
}

void light<pointLight>::scale(const glm::vec3 & scale)
{
    m_scale = scale;
    updateViewMatrix();
}

void light<pointLight>::setPosition(const glm::vec3& translate)
{
    m_translate = translate;
    updateViewMatrix();
}

void light<pointLight>::updateViewMatrix()
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

void light<pointLight>::rotateX(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    m_rotateX = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax)) * m_rotateX;
    m_rotate = m_rotateX * m_rotateY;
    updateViewMatrix();
}

void light<pointLight>::rotateY(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    m_rotateY = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax)) * m_rotateY;
    updateViewMatrix();
}

