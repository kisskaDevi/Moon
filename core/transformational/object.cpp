#include "object.h"
#include "core/operations.h"

#include "gltfmodel.h"

object::object()
{
    modelMatrix = glm::mat4x4(1.0f);
    m_globalTransform = glm::mat4x4(1.0f);
    m_translate = glm::vec3(0.0f,0.0f,0.0f);
    m_rotate = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_scale = glm::vec3(1.0f,1.0f,1.0f);

}

object::object(uint32_t modelCount, gltfModel** model)
{
    modelMatrix = glm::mat4x4(1.0f);
    m_globalTransform = glm::mat4x4(1.0f);
    m_translate = glm::vec3(0.0f,0.0f,0.0f);
    m_rotate = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_scale = glm::vec3(1.0f,1.0f,1.0f);

    this->pModel = model;
    this->modelCount = modelCount;
}

object::~object()
{

}

void object::destroyUniformBuffers(VkDevice* device)
{
    for(size_t i=0;i<uniformBuffers.size();i++)
    {
        if (uniformBuffers[i] != VK_NULL_HANDLE)
        {
            vkDestroyBuffer(*device, uniformBuffers.at(i), nullptr);
            vkFreeMemory(*device, uniformBuffersMemory.at(i), nullptr);
        }
    }
}

void object::destroyDescriptorPools(VkDevice* device)
{
    if (descriptorPool != VK_NULL_HANDLE){
        vkDestroyDescriptorPool(*device, descriptorPool, nullptr);
        descriptorPool = VK_NULL_HANDLE;
    }
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

void object::updateAnimation(uint32_t imageNumber)
{
    if(modelCount>1){
        if(pModel[imageNumber]->animations.size() > 0){
            if(!changeAnimationFlag){
                if (animationTimer > pModel[imageNumber]->animations[animationIndex].end)
                    animationTimer -= pModel[imageNumber]->animations[animationIndex].end;
                pModel[imageNumber]->updateAnimation(animationIndex, animationTimer);
            }else{
                pModel[imageNumber]->changeAnimation(animationIndex, newAnimationIndex, startTimer, animationTimer, changeAnimationTime);
                if(startTimer+changeAnimationTime<animationTimer){
                    changeAnimationFlag = false;
                    animationTimer = pModel[imageNumber]->animations[animationIndex+1].start;
                    animationIndex = newAnimationIndex;
                }
            }
        }
    }
}

void object::createUniformBuffers(VkPhysicalDevice* physicalDevice, VkDevice* device, uint32_t imageCount)
{
    uniformBuffers.resize(imageCount);
    uniformBuffersMemory.resize(imageCount);
    for (size_t i = 0; i < imageCount; i++){
        createBuffer(   physicalDevice,
                        device,
                        sizeof(UniformBuffer),
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        uniformBuffers[i],
                        uniformBuffersMemory[i]);
    }
}

void object::updateUniformBuffer(VkDevice* device, uint32_t currentImage)
{
    void* data;
    UniformBuffer ubo{};
        ubo.modelMatrix = modelMatrix;
        ubo.color = color;
    vkMapMemory(*device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
        memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(*device, uniformBuffersMemory[currentImage]);
}

void                            object::setModel(gltfModel** model3D)                    {this->pModel = model3D;}
void                            object::setVisibilityDistance(float visibilityDistance) {this->visibilityDistance=visibilityDistance;}
void                            object::setColor(const glm::vec4 &color)                {this->color = color;}
void                            object::setEnable(const bool& enable)                   {this->enable = enable;}

gltfModel*                      object::getModel(uint32_t index)                        {
    gltfModel* model;
    if(modelCount>1&&index>=modelCount){
        model = nullptr;
    }else if(modelCount==1){
        model = pModel[0];
    }else{
        model = pModel[index];
    }
    return model;
}
float                           object::getVisibilityDistance() const                   {return visibilityDistance;}
glm::vec4                       object::getColor()              const                   {return color;}
bool                            object::getEnable()             const                   {return enable;}

glm::mat4x4                     &object::ModelMatrix()                                  {return modelMatrix;}
glm::mat4x4                     &object::Transformation()                               {return m_globalTransform;}
glm::vec3                       &object::Translate()                                    {return m_translate;}
glm::quat                       &object::Rotate()                                       {return m_rotate;}
glm::vec3                       &object::Scale()                                        {return m_scale;}

VkDescriptorPool                &object::getDescriptorPool()                            {return descriptorPool;}
std::vector<VkDescriptorSet>    &object::getDescriptorSet()                             {return descriptors;}
std::vector<VkBuffer>           &object::getUniformBuffers()                            {return uniformBuffers;}


void                            object::setStencilEnable(const bool& enable)            {stencil.Enable = enable;}
void                            object::setStencilWidth(const float& width)             {stencil.Width = width;}
void                            object::setStencilColor(const glm::vec4& color)         {stencil.Color = color;}

bool                            object::getStencilEnable() const                        {return stencil.Enable;}
float                           object::getStencilWidth()  const                        {return stencil.Width;}
glm::vec4                       object::getStencilColor()  const                        {return stencil.Color;}

void                            object::setFirstPrimitive(uint32_t firstPrimitive)      {this->firstPrimitive = firstPrimitive;}
void                            object::setPrimitiveCount(uint32_t primitiveCount)      {this->primitiveCount = primitiveCount;}
void                            object::resetPrimitiveCount()                           {primitiveCount=0;}
void                            object::increasePrimitiveCount()                        {primitiveCount++;}

bool                            object::comparePrimitive(uint32_t primitive)            {return primitive>=firstPrimitive&&primitive<firstPrimitive+primitiveCount;}
uint32_t                        object::getFirstPrimitive() const                       {return firstPrimitive;}
uint32_t                        object::getPrimitiveCount() const                       {return primitiveCount;}
