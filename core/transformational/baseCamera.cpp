#include "baseCamera.h"
#include "operations.h"
#include "dualQuaternion.h"

#include <cstring>

baseCamera::baseCamera(){}

baseCamera::baseCamera(float angle, float aspect, float near, float far)
{
    glm::mat4x4 proj = glm::perspective(glm::radians(angle), aspect, near, far);
    proj[1][1] *= -1.0f;
    setProjMatrix(proj);
}

void baseCamera::recreate(float angle, float aspect, float near, float far)
{
    glm::mat4x4 proj = glm::perspective(glm::radians(angle), aspect, near, far);
    proj[1][1] *= -1.0f;
    setProjMatrix(proj);
}

baseCamera::~baseCamera(){}

void baseCamera::destroy(VkDevice device)
{
    destroyUniformBuffers(device, uniformBuffersHost);
    destroyUniformBuffers(device, uniformBuffersDevice);
}

void baseCamera::destroyUniformBuffers(VkDevice device, std::vector<buffer>& uniformBuffers)
{
    for(auto& buffer: uniformBuffers){
        buffer.destroy(device);
    }
    uniformBuffers.clear();
}

void baseCamera::updateUniformBuffersFlags(std::vector<buffer>& uniformBuffers)
{
    for (auto& buffer: uniformBuffers){
        buffer.updateFlag = true;
    }
}

void baseCamera::updateViewMatrix()
{
    dualQuaternion<float> dQuat = convert(rotation,translation);
    glm::mat<4,4,float,glm::defaultp> transformMatrix = convert(dQuat);

    viewMatrix = glm::inverse(globalTransformation) * glm::inverse(transformMatrix);

    updateUniformBuffersFlags(uniformBuffersHost);
}

void baseCamera::setProjMatrix(const glm::mat4 & proj)
{
    projMatrix = proj;

    updateUniformBuffersFlags(uniformBuffersHost);
}

void baseCamera::setGlobalTransform(const glm::mat4 & transform)
{
    globalTransformation = transform;
    updateViewMatrix();
}

void baseCamera::translate(const glm::vec3 & translate)
{
    translation += quaternion<float>(0.0f,translate);
    updateViewMatrix();
}

void baseCamera::rotate(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    rotation = convert(ang,ax)*rotation;
    updateViewMatrix();
}

void baseCamera::scale(const glm::vec3 &scale)
{
    static_cast<void>(scale);
}

void baseCamera::rotateX(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    rotationX = convert(ang,ax) * rotationX;
    rotation = rotationX * rotationY;
    updateViewMatrix();
}

void baseCamera::rotateY(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    rotationY = convert(ang,ax) * rotationY;
    rotation = rotationX * rotationY;
    updateViewMatrix();
}

void baseCamera::setPosition(const glm::vec3 & translate)
{
    translation = quaternion<float>(0.0f,translate);
    updateViewMatrix();
}

void baseCamera::setRotation(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    rotation = convert(ang,ax);
    updateViewMatrix();
}

void baseCamera::setRotation(const quaternion<float>& rotation)
{
    this->rotation = rotation;
    updateViewMatrix();
}

void baseCamera::setRotations(const quaternion<float>& quatX, const quaternion<float>& quatY)
{
    this->rotationX = quatX;
    this->rotationY = quatY;
}

glm::mat4x4 baseCamera::getProjMatrix() const
{
    return projMatrix;
}

glm::mat4x4 baseCamera::getViewMatrix() const
{
    return viewMatrix;
}

glm::vec3           baseCamera::getTranslation() const  {   return translation.vector();}
quaternion<float>   baseCamera::getRotationX()const     {   return rotationX;}
quaternion<float>   baseCamera::getRotationY()const     {   return rotationY;}

void baseCamera::createUniformBuffers(VkPhysicalDevice physicalDevice, VkDevice device, uint32_t imageCount)
{
    uniformBuffersHost.resize(imageCount);
    for (auto& buffer: uniformBuffersHost){
      Buffer::create(   physicalDevice,
                        device,
                        sizeof(UniformBufferObject),
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        &buffer.instance,
                        &buffer.memory);
      vkMapMemory(device, buffer.memory, 0, sizeof(UniformBufferObject), 0, &buffer.map);
    }
    uniformBuffersDevice.resize(imageCount);
    for (auto& buffer: uniformBuffersDevice){
      Buffer::create(   physicalDevice,
                        device,
                        sizeof(UniformBufferObject),
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        &buffer.instance,
                        &buffer.memory);
    }
}

void baseCamera::updateUniformBuffer(VkCommandBuffer commandBuffer, uint32_t frameNumber)
{
    if(uniformBuffersHost[frameNumber].updateFlag){
        UniformBufferObject baseUBO{};
            baseUBO.view = viewMatrix;
            baseUBO.proj = projMatrix;
            baseUBO.eyePosition = glm::vec4(translation.vector(), 1.0);
        std::memcpy(uniformBuffersHost[frameNumber].map, &baseUBO, sizeof(baseUBO));

        uniformBuffersHost[frameNumber].updateFlag = false;

        Buffer::copy(commandBuffer, sizeof(UniformBufferObject), uniformBuffersHost[frameNumber].instance, uniformBuffersDevice[frameNumber].instance);
    }
}

VkBuffer baseCamera::getBuffer(uint32_t index)const
{
    return uniformBuffersDevice[index].instance;
}

VkDeviceSize baseCamera::getBufferRange() const
{
    return sizeof(UniformBufferObject);
}
