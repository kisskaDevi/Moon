#include "baseCamera.h"
#include "operations.h"
#include "dualQuaternion.h"

#include <cstring>


baseCamera::baseCamera(){}

baseCamera::baseCamera(float angle, float aspect, float near)
{
    matrix<float,4,4> proj = perspective(radians(angle), aspect, near);
    setProjMatrix(proj);
}

baseCamera::baseCamera(float angle, float aspect, float near, float far)
{
    matrix<float,4,4> proj = perspective(radians(angle), aspect, near, far);
    setProjMatrix(proj);
}

void baseCamera::recreate(float angle, float aspect, float near, float far)
{
    matrix<float,4,4> proj = perspective(radians(angle), aspect, near, far);
    setProjMatrix(proj);
}

void baseCamera::recreate(float angle, float aspect, float near)
{
    matrix<float,4,4> proj = perspective(radians(angle), aspect, near);
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
    dualQuaternion<float> dQuat = convert(rotation, translation);
    matrix<float,4,4> transformMatrix = convert(dQuat);
    viewMatrix = inverse(matrix<float,4,4>(globalTransformation * transformMatrix));

    updateUniformBuffersFlags(uniformBuffersHost);
}

baseCamera& baseCamera::setProjMatrix(const matrix<float,4,4> & proj)
{
    projMatrix = proj;
    updateUniformBuffersFlags(uniformBuffersHost);
    return *this;
}

baseCamera& baseCamera::setGlobalTransform(const matrix<float,4,4> & transform)
{
    globalTransformation = transform;
    updateViewMatrix();
    return *this;
}

baseCamera& baseCamera::translate(const vector<float,3> & translate)
{
    translation += quaternion<float>(0.0f,translate);
    updateViewMatrix();
    return *this;
}

baseCamera& baseCamera::rotate(const float & ang ,const vector<float,3> & ax)
{
    rotation = convert(ang,vector<float,3>(normalize(ax)))*rotation;
    updateViewMatrix();
    return *this;
}

baseCamera& baseCamera::scale(const vector<float,3> &){
    return *this;
}

baseCamera& baseCamera::rotate(const quaternion<float>& quat)
{
    rotation = quat * rotation;
    updateViewMatrix();
    return *this;
}

baseCamera& baseCamera::rotateX(const float & ang ,const vector<float,3> & ax)
{
    rotationX = convert(ang,vector<float,3>(normalize(ax))) * rotationX;
    rotation = rotationY * rotationX;
    updateViewMatrix();
    return *this;
}

baseCamera& baseCamera::rotateY(const float & ang ,const vector<float,3> & ax)
{
    rotationY = convert(ang,vector<float,3>(normalize(ax))) * rotationY;
    rotation = rotationY * rotationX;
    updateViewMatrix();
    return *this;
}

baseCamera& baseCamera::setTranslation(const vector<float,3> & translate)
{
    translation = quaternion<float>(0.0f,translate);
    updateViewMatrix();
    return *this;
}

baseCamera& baseCamera::setRotation(const float & ang ,const vector<float,3> & ax)
{
    rotation = convert(ang,vector<float,3>(normalize(ax)));
    updateViewMatrix();
    return *this;
}

baseCamera& baseCamera::setRotation(const quaternion<float>& rotation)
{
    this->rotation = rotation;
    updateViewMatrix();
    return *this;
}

matrix<float,4,4> baseCamera::getProjMatrix() const
{
    return projMatrix;
}

matrix<float,4,4> baseCamera::getViewMatrix() const
{
    return viewMatrix;
}

vector<float,3>     baseCamera::getTranslation() const  {   return translation.vector();}
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
            baseUBO.view = transpose(viewMatrix);
            baseUBO.proj = transpose(projMatrix);
            baseUBO.eyePosition = vector<float,4>(translation.vector()[0], translation.vector()[1], translation.vector()[2], 1.0);
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
