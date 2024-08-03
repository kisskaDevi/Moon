#ifndef BASECAMERA_H
#define BASECAMERA_H

#include <vulkan.h>

#include "camera.h"
#include "transformational.h"
#include "quaternion.h"
#include "buffer.h"

namespace moon::interfaces {

class BaseCamera : public Camera
{
private:
    utils::UniformBuffer uniformBuffer;

    void create(const utils::PhysicalDevice& device, uint32_t imageCount) override;
    void update(uint32_t frameNumber, VkCommandBuffer commandBuffer) override;

public:
    BaseCamera(void* hostData, size_t hostDataSize);
    utils::Buffers& buffers() override;
};

}

namespace moon::transformational {

struct UniformBufferObject{
    alignas(16) math::Matrix<float,4,4>   view{ 1.0f };
    alignas(16) math::Matrix<float,4,4>   proj{ 1.0f };
};

class Camera : public Transformational
{
private:
    UniformBufferObject buffer;

    math::Matrix<float,4,4> globalTransformation{1.0f};
    math::Quaternion<float> translation{0.0f,0.0f,0.0f,0.0f};
    math::Quaternion<float> rotation{1.0f,0.0f,0.0f,0.0f};

    std::unique_ptr<interfaces::Camera> pCamera;

    Camera& updateViewMatrix();

public:
    Camera();
    Camera(const float& angle, const float& aspect, const float& n, const float& f = std::numeric_limits<float>::max());

    Camera& setGlobalTransform(const math::Matrix<float,4,4> & transform) override;
    Camera& translate(const math::Vector<float,3> & translate) override;
    Camera& rotate(const float & ang ,const math::Vector<float,3> & ax) override;
    Camera& scale(const math::Vector<float,3> & scale) override;

    Camera& rotateX(const float& ang);
    Camera& rotateY(const float& ang);

    Camera& setProjMatrix(const math::Matrix<float,4,4> & proj);
    Camera& setTranslation(const math::Vector<float,3> & translate);
    Camera& setRotation(const math::Quaternion<float>& rotation);

    math::Vector<float,3>   getTranslation() const;
    math::Quaternion<float> getRotation()    const;
    math::Matrix<float,4,4> getProjMatrix()  const;
    math::Matrix<float,4,4> getViewMatrix()  const;

    operator interfaces::Camera*() const;
};

}
#endif // BASECAMERA_H
