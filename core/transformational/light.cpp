#include "light.h"
#include "object.h"
#include "core/operations.h"
#include "core/graphics/shadowGraphics.h"
#include "gltfmodel.h"

light<spotLight>::light(VkApplication *app, uint32_t type) : app(app), type(type)
{
    m_scale = glm::vec3(1.0f,1.0f,1.0f);
    m_globalTransform = glm::mat4x4(1.0f);
    m_translate = glm::vec3(0.0f,0.0f,0.0f);
    m_rotate = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_rotateX = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_rotateY = glm::quat(1.0f,0.0f,0.0f,0.0f);
    viewMatrix = glm::mat4x4(1.0f);
    modelMatrix = glm::mat4x4(1.0f);

    uint32_t imageCount = app->getImageCount();
    uniformBuffers.resize(imageCount);
    uniformBuffersMemory.resize(imageCount);
    for (size_t i = 0; i < imageCount; i++){
        createBuffer(   app,
                        sizeof(LightBufferObject),
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        uniformBuffers[i],
                        uniformBuffersMemory[i]);
    }
}

light<spotLight>::~light(){}

void light<spotLight>::cleanup()
{
    vkDestroyDescriptorPool(app->getDevice(), descriptorPool, nullptr);

    for(size_t i=0;i<uniformBuffers.size();i++)
    {
        if (uniformBuffers.at(i) != VK_NULL_HANDLE)
        {
            vkDestroyBuffer(app->getDevice(), uniformBuffers.at(i), nullptr);
            vkFreeMemory(app->getDevice(), uniformBuffersMemory.at(i), nullptr);
        }
    }

    if(enableShadow)
        shadow->destroy();
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

void light<spotLight>::createLightPVM(const glm::mat4x4 & projection)
{
    projectionMatrix = projection;
}

void light<spotLight>::createShadow(uint32_t imageCount)
{
    enableShadow = true;
    shadow = new shadowGraphics(app,imageCount,shadowExtent);
    shadow->createShadow();
}

void light<spotLight>::updateLightBuffer(uint32_t frameNumber)
{
    LightBufferObject buffer{};
        buffer.proj = projectionMatrix;
        buffer.view = viewMatrix;
        buffer.projView = projectionMatrix * viewMatrix;
        buffer.position = modelMatrix * glm::vec4(0.0f,0.0f,0.0f,1.0f);
        buffer.lightColor = lightColor;
        buffer.type = type;
    void* data;
    vkMapMemory(app->getDevice(), uniformBuffersMemory[frameNumber], 0, sizeof(buffer), 0, &data);
        memcpy(data, &buffer, sizeof(buffer));
    vkUnmapMemory(app->getDevice(), uniformBuffersMemory[frameNumber]);
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
bool                            light<spotLight>::getShadowEnable() const{return enableShadow;}
bool                            light<spotLight>::getScatteringEnable() const{return enableScattering;}

shadowGraphics                  *light<spotLight>::getShadow(){return shadow;}
texture                         *light<spotLight>::getTexture(){return tex;}

//======================================================================================================================//
//============//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//============//
//============//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//============//
//============//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//==//============//
//======================================================================================================================//

light<pointLight>::light(VkApplication *app, std::vector<light<spotLight> *> & lightSource) : lightSource(lightSource)
{
    m_scale = glm::vec3(1.0f,1.0f,1.0f);
    m_globalTransform = glm::mat4x4(1.0f);
    m_translate = glm::vec3(0.0f,0.0f,0.0f);
    m_rotate = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_rotateX = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_rotateY = glm::quat(1.0f,0.0f,0.0f,0.0f);
    number = lightSource.size();

    glm::mat4x4 Proj = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 100.0f);
    Proj[1][1] *= -1;

    int index = number;
    lightSource.push_back(new light<spotLight>(app,lightType::point));
    lightSource.at(index)->createLightPVM(Proj);
    lightSource.at(index)->rotate(glm::radians(90.0f),glm::vec3(1.0f,0.0f,0.0f));
    lightSource.at(index)->setLightColor(glm::vec4(1.0f,0.0f,0.0f,1.0f));
    app->addlightSource(lightSource.at(index));

    index++;
    lightSource.push_back(new light<spotLight>(app,lightType::point));
    lightSource.at(index)->createLightPVM(Proj);
    lightSource.at(index)->rotate(glm::radians(-90.0f),glm::vec3(1.0f,0.0f,0.0f));
    lightSource.at(index)->setLightColor(glm::vec4(0.0f,1.0f,0.0f,1.0f));
    app->addlightSource(lightSource.at(index));

    index++;
    lightSource.push_back(new light<spotLight>(app,lightType::point));
    lightSource.at(index)->createLightPVM(Proj);
    lightSource.at(index)->setLightColor(glm::vec4(0.0f,0.0f,1.0f,1.0f));
    app->addlightSource(lightSource.at(index));

    index++;
    lightSource.push_back(new light<spotLight>(app,lightType::point));
    lightSource.at(index)->createLightPVM(Proj);
    lightSource.at(index)->rotate(glm::radians(90.0f),glm::vec3(0.0f,1.0f,0.0f));
    lightSource.at(index)->setLightColor(glm::vec4(0.3f,0.6f,0.9f,1.0f));
    app->addlightSource(lightSource.at(index));

    index++;
    lightSource.push_back(new light<spotLight>(app,lightType::point));
    lightSource.at(index)->createLightPVM(Proj);
    lightSource.at(index)->rotate(glm::radians(-90.0f),glm::vec3(0.0f,1.0f,0.0f));
    lightSource.at(index)->setLightColor(glm::vec4(0.6f,0.9f,0.3f,1.0f));
    app->addlightSource(lightSource.at(index));

    index++;
    lightSource.push_back(new light<spotLight>(app,lightType::point));
    lightSource.at(index)->createLightPVM(Proj);
    lightSource.at(index)->rotate(glm::radians(180.0f),glm::vec3(1.0f,0.0f,0.0f));
    lightSource.at(index)->setLightColor(glm::vec4(0.9f,0.3f,0.6f,1.0f));
    app->addlightSource(lightSource.at(index));
}

light<pointLight>::~light(){}

glm::vec4       light<pointLight>::getLightColor() const {return lightColor;}
uint32_t        light<pointLight>::getNumber() const {return number;}
glm::vec3       light<pointLight>::getTranslate() const {return m_translate;}

void light<pointLight>::setLightColor(const glm::vec4 &color)
{
    this->lightColor = color;
    for(uint32_t i=number;i<number+6;i++)
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

    for(uint32_t i=number;i<number+6;i++)
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

