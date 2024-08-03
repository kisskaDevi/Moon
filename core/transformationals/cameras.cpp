#include "cameras.h"

#include "operations.h"
#include "dualQuaternion.h"
#include "device.h"

#include <cstring>

namespace moon::interfaces {

BaseCamera::BaseCamera(void* hostData, size_t hostDataSize)
    : uniformBuffer(hostData, hostDataSize) {}

void BaseCamera::update(uint32_t frameNumber, VkCommandBuffer commandBuffer) {
    uniformBuffer.update(frameNumber, commandBuffer);
}

void BaseCamera::create(const utils::PhysicalDevice& device, uint32_t imageCount) {
    uniformBuffer = utils::UniformBuffer(device, imageCount, uniformBuffer.host, uniformBuffer.size);
}

utils::Buffers& BaseCamera::buffers() {
    return uniformBuffer.device;
}

}

namespace moon::transformational {

Camera::Camera() {
    pCamera = std::make_unique<interfaces::BaseCamera>(&buffer, sizeof(buffer));
}

Camera::Camera(const float& angle, const float& aspect, const float& near, const float& far) : Camera() {
    setProjMatrix(math::perspective(math::radians(angle), aspect, near, far));
}

Camera& Camera::updateViewMatrix() {
    math::Matrix<float,4,4> transformMatrix = convert(convert(rotation, translation));
    buffer.view = transpose(inverse(math::Matrix<float,4,4>(globalTransformation * transformMatrix)));
    utils::raiseFlags(pCamera->buffers());
    return *this;
}

Camera& Camera::setProjMatrix(const math::Matrix<float,4,4> & proj) {
    buffer.proj = transpose(proj);
    utils::raiseFlags(pCamera->buffers());
    return *this;
}

Camera& Camera::setGlobalTransform(const math::Matrix<float,4,4> & transform) {
    globalTransformation = transform;
    return updateViewMatrix();
}

Camera& Camera::translate(const math::Vector<float,3> & translate) {
    translation += math::Quaternion<float>(0.0f,translate);
    return updateViewMatrix();
}

Camera& Camera::rotate(const float & ang ,const math::Vector<float,3> & ax) {
    rotation = convert(ang, math::Vector<float,3>(normalize(ax))) * rotation;
    return updateViewMatrix();
}

Camera& Camera::scale(const math::Vector<float,3> &){
    return *this;
}

Camera& Camera::rotateX(const float& ang) {
    const math::Vector<float, 3> ax(1.0f, 0.0f, 0.0f);
    rotation = rotation * convert(ang, math::Vector<float, 3>(normalize(ax)));
    return updateViewMatrix();
}

Camera& Camera::rotateY(const float& ang) {
    const math::Vector<float, 3> ax(0.0f, 0.0f, 1.0f);
    rotation = convert(ang, math::Vector<float, 3>(normalize(ax))) * rotation;
    return updateViewMatrix();
}

Camera& Camera::setTranslation(const math::Vector<float,3> & trans) {
    translation = math::Quaternion<float>(0.0f, trans);
    return updateViewMatrix();
}

Camera& Camera::setRotation(const math::Quaternion<float>& rot) {
    rotation = rot;
    return updateViewMatrix();
}

math::Matrix<float,4,4> Camera::getProjMatrix()  const { return transpose(buffer.proj);}
math::Matrix<float,4,4> Camera::getViewMatrix()  const { return transpose(buffer.view);}
math::Vector<float,3>   Camera::getTranslation() const { return translation.im();}
math::Quaternion<float> Camera::getRotation()    const { return rotation; }

Camera::operator interfaces::Camera* () const {
    return pCamera.get();
}

}
