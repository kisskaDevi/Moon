#include "baseCamera.h"
#include "operations.h"
#include "dualQuaternion.h"
#include "device.h"

#include <cstring>

namespace moon::transformational {

BaseCamera::BaseCamera(){}

BaseCamera::BaseCamera(float angle, float aspect, float near)
{
    moon::math::Matrix<float,4,4> proj = moon::math::perspective(moon::math::radians(angle), aspect, near);
    setProjMatrix(proj);
}

BaseCamera::BaseCamera(float angle, float aspect, float near, float far)
{
    moon::math::Matrix<float,4,4> proj = moon::math::perspective(moon::math::radians(angle), aspect, near, far);
    setProjMatrix(proj);
}

void BaseCamera::recreate(float angle, float aspect, float near, float far)
{
    moon::math::Matrix<float,4,4> proj = moon::math::perspective(moon::math::radians(angle), aspect, near, far);
    setProjMatrix(proj);
}

void BaseCamera::recreate(float angle, float aspect, float near)
{
    moon::math::Matrix<float,4,4> proj = moon::math::perspective(moon::math::radians(angle), aspect, near);
    setProjMatrix(proj);
}

BaseCamera::~BaseCamera(){
    BaseCamera::destroy(device);
}

void BaseCamera::destroy(VkDevice device) {
    created = false;
}

void updateUniformBuffersFlags(moon::utils::Buffers& uniformBuffers)
{
    for (auto& buffer: uniformBuffers){
        buffer.raiseFlag();
    }
}

void BaseCamera::updateViewMatrix()
{
    moon::math::DualQuaternion<float> dQuat = convert(rotation, translation);
    moon::math::Matrix<float,4,4> transformMatrix = convert(dQuat);
    viewMatrix = inverse(moon::math::Matrix<float,4,4>(globalTransformation * transformMatrix));

    updateUniformBuffersFlags(uniformBuffersHost);
}

BaseCamera& BaseCamera::setProjMatrix(const moon::math::Matrix<float,4,4> & proj)
{
    projMatrix = proj;
    updateUniformBuffersFlags(uniformBuffersHost);
    return *this;
}

BaseCamera& BaseCamera::setGlobalTransform(const moon::math::Matrix<float,4,4> & transform)
{
    globalTransformation = transform;
    updateViewMatrix();
    return *this;
}

BaseCamera& BaseCamera::translate(const moon::math::Vector<float,3> & translate)
{
    translation += moon::math::Quaternion<float>(0.0f,translate);
    updateViewMatrix();
    return *this;
}

BaseCamera& BaseCamera::rotate(const float & ang ,const moon::math::Vector<float,3> & ax)
{
    rotation = convert(ang, moon::math::Vector<float,3>(normalize(ax))) * rotation;
    updateViewMatrix();
    return *this;
}

BaseCamera& BaseCamera::scale(const moon::math::Vector<float,3> &){
    return *this;
}

BaseCamera& BaseCamera::rotate(const moon::math::Quaternion<float>& quat)
{
    rotation = quat * rotation;
    updateViewMatrix();
    return *this;
}

BaseCamera& BaseCamera::rotateX(const float & ang ,const moon::math::Vector<float,3> & ax)
{
    rotationX = convert(ang, moon::math::Vector<float,3>(normalize(ax))) * rotationX;
    rotation = rotationY * rotationX;
    updateViewMatrix();
    return *this;
}

BaseCamera& BaseCamera::rotateY(const float & ang ,const moon::math::Vector<float,3> & ax)
{
    rotationY = convert(ang, moon::math::Vector<float,3>(normalize(ax))) * rotationY;
    rotation = rotationY * rotationX;
    updateViewMatrix();
    return *this;
}

BaseCamera& BaseCamera::setTranslation(const moon::math::Vector<float,3> & translate)
{
    translation = moon::math::Quaternion<float>(0.0f,translate);
    updateViewMatrix();
    return *this;
}

BaseCamera& BaseCamera::setRotation(const float & ang ,const moon::math::Vector<float,3> & ax)
{
    rotation = convert(ang, moon::math::Vector<float,3>(normalize(ax)));
    updateViewMatrix();
    return *this;
}

BaseCamera& BaseCamera::setRotation(const moon::math::Quaternion<float>& rotation)
{
    this->rotation = rotation;
    updateViewMatrix();
    return *this;
}

moon::math::Matrix<float,4,4> BaseCamera::getProjMatrix() const
{
    return projMatrix;
}

moon::math::Matrix<float,4,4> BaseCamera::getViewMatrix() const
{
    return viewMatrix;
}

moon::math::Vector<float,3>     BaseCamera::getTranslation() const  {   return translation.im();}
moon::math::Quaternion<float>   BaseCamera::getRotationX()const     {   return rotationX;}
moon::math::Quaternion<float>   BaseCamera::getRotationY()const     {   return rotationY;}

void BaseCamera::createUniformBuffers(VkPhysicalDevice physicalDevice, VkDevice device, uint32_t imageCount)
{
    uniformBuffersHost.resize(imageCount);
    for (auto& buffer: uniformBuffersHost){
        buffer.create(physicalDevice, device, sizeof(UniformBufferObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        moon::utils::Memory::instance().nameMemory(buffer, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", baseCamera::createUniformBuffers, uniformBuffersHost " + std::to_string(&buffer - &uniformBuffersHost[0]));
    }
    uniformBuffersDevice.resize(imageCount);
    for (auto& buffer: uniformBuffersDevice){
        buffer.create(physicalDevice, device, sizeof(UniformBufferObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        moon::utils::Memory::instance().nameMemory(buffer, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", baseCamera::createUniformBuffers, uniformBuffersDevice " + std::to_string(&buffer - &uniformBuffersDevice[0]));
    }
}

void BaseCamera::update(uint32_t frameNumber, VkCommandBuffer commandBuffer)
{
    if(auto& buffer = uniformBuffersHost[frameNumber]; buffer.dropFlag()){
        UniformBufferObject baseUBO{};
            baseUBO.view = transpose(viewMatrix);
            baseUBO.proj = transpose(projMatrix);
            baseUBO.eyePosition = moon::math::Vector<float,4>(translation.im()[0], translation.im()[1], translation.im()[2], 1.0);
        buffer.copy(&baseUBO);

        moon::utils::buffer::copy(commandBuffer, sizeof(UniformBufferObject), buffer, uniformBuffersDevice[frameNumber]);
    }
}

void BaseCamera::create(const moon::utils::PhysicalDevice& device, uint32_t imageCount)
{
    if(!created){
        createUniformBuffers(device.instance,device.getLogical(),imageCount);
        created = true;
        this->device = device.getLogical();
    }
}

const moon::utils::Buffers& BaseCamera::getBuffers() const {
    return uniformBuffersDevice;
}

}
