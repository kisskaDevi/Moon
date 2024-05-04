#include "baseCamera.h"
#include "operations.h"
#include "dualQuaternion.h"
#include "device.h"

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

baseCamera::~baseCamera(){
    baseCamera::destroy(device);
}

void baseCamera::destroy(VkDevice device)
{
    uniformBuffersHost.destroy(device);
    uniformBuffersDevice.destroy(device);
    created = false;
}

void updateUniformBuffersFlags(moon::utils::Buffers& uniformBuffers)
{
    for (auto& buffer: uniformBuffers.instances){
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

vector<float,3>     baseCamera::getTranslation() const  {   return translation.im();}
quaternion<float>   baseCamera::getRotationX()const     {   return rotationX;}
quaternion<float>   baseCamera::getRotationY()const     {   return rotationY;}

void baseCamera::createUniformBuffers(VkPhysicalDevice physicalDevice, VkDevice device, uint32_t imageCount)
{
    uniformBuffersHost.create(physicalDevice,
                              device,
                              sizeof(UniformBufferObject),
                              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                              imageCount);
    uniformBuffersHost.map(device);
    for (auto& buffer: uniformBuffersHost.instances){
        moon::utils::Memory::instance().nameMemory(  buffer.memory,
                                        std::string(__FILE__) +
                                        " in line " + std::to_string(__LINE__) +
                                        ", baseCamera::createUniformBuffers, uniformBuffersHost " +
                                        std::to_string(&buffer - &uniformBuffersHost.instances[0]));
    }
    uniformBuffersDevice.create(physicalDevice,
                                device,
                                sizeof(UniformBufferObject),
                                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                imageCount);
    for (auto& buffer: uniformBuffersDevice.instances){
        moon::utils::Memory::instance().nameMemory(buffer.memory,
                                    std::string(__FILE__) +
                                    " in line " + std::to_string(__LINE__) +
                                    ", baseCamera::createUniformBuffers, uniformBuffersDevice " +
                                    std::to_string(&buffer - &uniformBuffersDevice.instances[0]));
    }
}

void baseCamera::update(uint32_t frameNumber, VkCommandBuffer commandBuffer)
{
    if(uniformBuffersHost.instances[frameNumber].updateFlag){
        UniformBufferObject baseUBO{};
            baseUBO.view = transpose(viewMatrix);
            baseUBO.proj = transpose(projMatrix);
            baseUBO.eyePosition = vector<float,4>(translation.im()[0], translation.im()[1], translation.im()[2], 1.0);
        std::memcpy(uniformBuffersHost.instances[frameNumber].map, &baseUBO, sizeof(baseUBO));

        uniformBuffersHost.instances[frameNumber].updateFlag = false;

        moon::utils::buffer::copy(commandBuffer, sizeof(UniformBufferObject), uniformBuffersHost.instances[frameNumber].instance, uniformBuffersDevice.instances[frameNumber].instance);
    }
}

void baseCamera::create(moon::utils::PhysicalDevice device, uint32_t imageCount)
{
    if(!created){
        createUniformBuffers(device.instance,device.getLogical(),imageCount);
        created = true;
        this->device = device.getLogical();
    }
}

const moon::utils::Buffers& baseCamera::getBuffers() const {
    return uniformBuffersDevice;
}
