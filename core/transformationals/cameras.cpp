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

Camera& Camera::update() {
    math::Matrix<float,4,4> transformMatrix = convert(convert(m_rotation, m_translation));
    buffer.view = transpose(inverse(math::Matrix<float,4,4>(m_globalTransformation * transformMatrix)));
    utils::raiseFlags(pCamera->buffers());
    return *this;
}

Camera& Camera::setProjMatrix(const math::Matrix<float,4,4> & proj) {
    buffer.proj = transpose(proj);
    utils::raiseFlags(pCamera->buffers());
    return *this;
}

DEFAULT_TRANSFORMATIONAL_DEFINITION(Camera)
DEFAULT_TRANSFORMATIONAL_GETTERS_DEFINITION(Camera)

Camera& Camera::rotateX(const float& ang) {
    const math::Vector<float, 3> ax(1.0f, 0.0f, 0.0f);
    m_rotation = m_rotation * convert(ang, math::Vector<float, 3>(normalize(ax)));
    return update();
}

Camera& Camera::rotateY(const float& ang) {
    const math::Vector<float, 3> ax(0.0f, 0.0f, 1.0f);
    m_rotation = convert(ang, math::Vector<float, 3>(normalize(ax))) * m_rotation;
    return update();
}

math::Matrix<float,4,4> Camera::getProjMatrix()  const { return transpose(buffer.proj);}
math::Matrix<float,4,4> Camera::getViewMatrix()  const { return transpose(buffer.view);}

Camera::operator interfaces::Camera* () const {
    return pCamera.get();
}

}
