#include "camera.h"
#include "core/operations.h"
#include "libs/dualQuaternion.h"

camera::camera(){}

camera::~camera(){}

void camera::updateViewMatrix()
{
    dualQuaternion<float> dQuat = convert(rotation,translation);
    glm::mat<4,4,float,glm::defaultp> transformMatrix = convert(dQuat);

    viewMatrix = glm::inverse(globalTransformation) * glm::inverse(transformMatrix);

    for (auto& buffer: uniformBuffers){
        buffer.updateFlag = true;
    }
}

void camera::setProjMatrix(const glm::mat4 & proj)
{
    projMatrix = proj;

    for (auto& buffer: uniformBuffers){
        buffer.updateFlag = true;
    }
}

void camera::setGlobalTransform(const glm::mat4 & transform)
{
    globalTransformation = transform;
    updateViewMatrix();
}

void camera::translate(const glm::vec3 & translate)
{
    translation += quaternion<float>(0.0f,translate);
    updateViewMatrix();
}

void camera::rotate(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    rotation = convert(ang,ax)*rotation;
    updateViewMatrix();
}

void camera::scale(const glm::vec3 &scale)
{
    static_cast<void>(scale);
}

void camera::rotateX(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    rotationX = convert(ang,ax) * rotationX;
    rotation = rotationX * rotationY;
    updateViewMatrix();
}

void camera::rotateY(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    rotationY = convert(ang,ax) * rotationY;
    rotation = rotationX * rotationY;
    updateViewMatrix();
}

void camera::setPosition(const glm::vec3 & translate)
{
    translation = quaternion<float>(0.0f,translate);
    updateViewMatrix();
}

void camera::setRotation(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    rotation = convert(ang,ax);
    updateViewMatrix();
}

void    camera::setRotation(const quaternion<float>& rotation)
{
    this->rotation = rotation;
    updateViewMatrix();
}

void    camera::setRotations(const quaternion<float>& quatX, const quaternion<float>& quatY)
{
    this->rotationX = quatX;
    this->rotationY = quatY;
}

glm::mat4x4 camera::getProjMatrix() const
{
    return projMatrix;
}

glm::mat4x4 camera::getViewMatrix() const
{
    return viewMatrix;
}

glm::vec3           camera::getTranslation() const  {   return translation.vector();}
quaternion<float>   camera::getRotationX()const     {   return rotationX;}
quaternion<float>   camera::getRotationY()const     {   return rotationY;}

void camera::destroyUniformBuffers(VkDevice* device)
{
    for(auto& buffer: uniformBuffers){
        if(buffer.instance) vkDestroyBuffer(*device, buffer.instance, nullptr);
        if(buffer.memory)   vkFreeMemory(*device, buffer.memory, nullptr);
    }
}

void camera::createUniformBuffers(VkPhysicalDevice* physicalDevice, VkDevice* device, uint32_t imageCount)
{
    uniformBuffers.resize(imageCount);
    for (auto& buffer: uniformBuffers){
        createBuffer(   physicalDevice,
                        device,
                        sizeof(UniformBufferObject),
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        buffer.instance,
                        buffer.memory);
    }
}

void camera::updateUniformBuffer(VkDevice* device, uint32_t currentImage)
{
    if(void* data; uniformBuffers[currentImage].updateFlag){
        UniformBufferObject baseUBO{};
            baseUBO.view = viewMatrix;
            baseUBO.proj = projMatrix;
            baseUBO.eyePosition = glm::vec4(translation.vector(), 1.0);
        vkMapMemory(*device, uniformBuffers[currentImage].memory, 0, sizeof(baseUBO), 0, &data);
            memcpy(data, &baseUBO, sizeof(baseUBO));
        vkUnmapMemory(*device, uniformBuffers[currentImage].memory);

        uniformBuffers[currentImage].updateFlag = false;
    }
}

VkBuffer camera::getBuffer(uint32_t index)const
{
    return uniformBuffers[index].instance;
}
