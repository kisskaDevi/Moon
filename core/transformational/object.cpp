#include "object.h"
#include "core/operations.h"

#include "gltfmodel.h"

object::object(VkApplication *app) : app(app)
{
    modelMatrix = glm::mat4x4(1.0f);
    m_globalTransform = glm::mat4x4(1.0f);
    m_translate = glm::vec3(0.0f,0.0f,0.0f);
    m_rotate = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_scale = glm::vec3(1.0f,1.0f,1.0f);

    model = nullptr;

    uint32_t imageCount = app->getImageCount();
    createUniformBuffers(imageCount);
}

object::object(VkApplication *app, objectInfo info): app(app)
{
    modelMatrix = glm::mat4x4(1.0f);
    m_globalTransform = glm::mat4x4(1.0f);
    m_translate = glm::vec3(0.0f,0.0f,0.0f);
    m_rotate = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_scale = glm::vec3(1.0f,1.0f,1.0f);

    model = info.model;
    emptyTexture = info.emptyTexture;

    uint32_t imageCount = app->getImageCount();
    createUniformBuffers(imageCount);
}

object::object(VkApplication *app, gltfModel* model3D) : app(app), model(model3D)
{
    modelMatrix = glm::mat4x4(1.0f);
    m_globalTransform = glm::mat4x4(1.0f);
    m_translate = glm::vec3(0.0f,0.0f,0.0f);
    m_rotate = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_scale = glm::vec3(1.0f,1.0f,1.0f);

    uint32_t imageCount = app->getImageCount();
    createUniformBuffers(imageCount);
}

object::~object()
{

}

void object::destroyUniformBuffers()
{
    for(size_t i=0;i<uniformBuffers.size();i++)
    {
        if (uniformBuffers.at(i) != VK_NULL_HANDLE)
        {
            vkDestroyBuffer(app->getDevice(), uniformBuffers.at(i), nullptr);
            vkFreeMemory(app->getDevice(), uniformBuffersMemory.at(i), nullptr);
        }
    }
}

void object::destroyDescriptorPools()
{
    vkDestroyDescriptorPool(app->getDevice(), descriptorPool, nullptr);
}

void object::setGlobalTransform(const glm::mat4x4 & transform)
{
    m_globalTransform = transform;
    updateModelMatrix();
}

void object::translate(const glm::vec3 & translate)
{
    m_translate += translate;
    updateModelMatrix();
}

void object::setPosition(const glm::vec3& translate)
{
    m_translate = translate;
    updateModelMatrix();
}

void object::rotate(const float & ang ,const glm::vec3 & ax)
{
    m_rotate = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax))*m_rotate;
    updateModelMatrix();
}

void object::scale(const glm::vec3 & scale)
{
    m_scale = scale;
    updateModelMatrix();
}

void object::updateModelMatrix()
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

void object::updateAnimation()
{
    if(getModel()->animations.size() > 0){
        if(!changeAnimationFlag){
            if (animationTimer > getModel()->animations[animationIndex].end)
                animationTimer -= getModel()->animations[animationIndex].end;

            getModel()->updateAnimation(animationIndex, animationTimer);
        }else{
            getModel()->changeAnimation(animationIndex, newAnimationIndex, startTimer, animationTimer, changeAnimationTime);
            if(startTimer+changeAnimationTime<animationTimer){
                changeAnimationFlag = false;
                animationTimer = getModel()->animations[animationIndex+1].start;
                animationIndex = newAnimationIndex;
            }
        }
    }
}

void object::createUniformBuffers(uint32_t imageCount)
{
    uniformBuffers.resize(imageCount);
    uniformBuffersMemory.resize(imageCount);
    for (size_t i = 0; i < imageCount; i++){
        createBuffer(app,sizeof(UniformBuffer), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);}
}

void object::updateUniformBuffer(uint32_t currentImage)
{
    void* data;
    UniformBuffer ubo{};
        ubo.modelMatrix = modelMatrix;
        ubo.color = color;
    vkMapMemory(app->getDevice(), uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
        memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(app->getDevice(), uniformBuffersMemory[currentImage]);
}

void                            object::setVisibilityDistance(float visibilityDistance){this->visibilityDistance=visibilityDistance;}
void                            object::setColor(const glm::vec4 &color){this->color = color;}
void                            object::setEmptyTexture(texture* emptyTexture){this->emptyTexture = emptyTexture;}
void                            object::setEnable(const bool& enable){this->enable = enable;}

gltfModel*                      object::getModel(){return model;}
float                           object::getVisibilityDistance(){return visibilityDistance;}
glm::vec4                       object::getColor(){return color;}

glm::mat4x4                     &object::ModelMatrix(){return modelMatrix;}
glm::mat4x4                     &object::Transformation(){return m_globalTransform;}
glm::vec3                       &object::Translate(){return m_translate;}
glm::quat                       &object::Rotate(){return m_rotate;}
glm::vec3                       &object::Scale(){return m_scale;}

VkDescriptorPool                &object::getDescriptorPool(){return descriptorPool;}
std::vector<VkDescriptorSet>    &object::getDescriptorSet(){return descriptors;}
std::vector<VkBuffer>           &object::getUniformBuffers(){return uniformBuffers;}

bool                            &object::getEnable(){return enable;}
