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
    matrix<float,4,4> transformMatrix = convert(dQuat);

    glm::mat<4,4,float,glm::defaultp> glmTransformMatrix;
    for(uint32_t i=0;i<4;i++){
        for(uint32_t j=0;j<4;j++){
            glmTransformMatrix[i][j] = transformMatrix[i][j];
        }
    }

    viewMatrix = glm::inverse(globalTransformation * glmTransformMatrix);

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
    translation += quaternion<float>(0.0f,vector<float,3>(translate[0],translate[1],translate[2]));
    updateViewMatrix();
}

void baseCamera::rotate(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    rotation = convert(ang,vector<float,3>(ax[0],ax[1],ax[2]))*rotation;
    updateViewMatrix();
}

void baseCamera::scale(const glm::vec3 &){}

void baseCamera::rotate(const quaternion<float>& quat)
{
    rotation = quat * rotation;
    updateViewMatrix();
}

void baseCamera::rotateX(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    rotationX = convert(ang,vector<float,3>(ax[0],ax[1],ax[2])) * rotationX;
    rotation = rotationX * rotationY;
    updateViewMatrix();
}

void baseCamera::rotateY(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    rotationY = convert(ang,vector<float,3>(ax[0],ax[1],ax[2])) * rotationY;
    rotation = rotationX * rotationY;
    updateViewMatrix();
}

void baseCamera::setPosition(const glm::vec3 & translate)
{
    translation = quaternion<float>(0.0f,vector<float,3>(translate[0],translate[1],translate[2]));
    updateViewMatrix();
}

void baseCamera::setRotation(const float & ang ,const glm::vec3 & ax)
{
    glm::normalize(ax);
    rotation = convert(ang,vector<float,3>(ax[0],ax[1],ax[2]));
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

glm::vec3           baseCamera::getTranslation() const  {   return glm::vec3(translation.vector()[0],translation.vector()[1],translation.vector()[2]);}
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
            baseUBO.eyePosition = glm::vec4(glm::vec3(translation.vector()[0],translation.vector()[1],translation.vector()[2]), 1.0);
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
