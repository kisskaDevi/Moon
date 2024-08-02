#ifndef BASECAMERA_H
#define BASECAMERA_H

#include <vulkan.h>

#include "camera.h"
#include "transformational.h"
#include "quaternion.h"
#include "buffer.h"

namespace moon::transformational {

struct UniformBufferObject{
    alignas(16) math::Matrix<float,4,4>   view;
    alignas(16) math::Matrix<float,4,4>   proj;
    alignas(16) math::Vector<float,4>     eyePosition;
};

class BaseCamera : public Transformational, public interfaces::Camera
{
private:
    math::Matrix<float,4,4>       projMatrix{1.0f};
    math::Matrix<float,4,4>       viewMatrix{1.0f};
    math::Matrix<float,4,4>       globalTransformation{1.0f};

    math::Quaternion<float>       translation{0.0f,0.0f,0.0f,0.0f};
    math::Quaternion<float>       rotation{1.0f,0.0f,0.0f,0.0f};
    math::Quaternion<float>       rotationX{1.0f,0.0f,0.0f,0.0f};
    math::Quaternion<float>       rotationY{1.0f,0.0f,0.0f,0.0f};

protected:
    const utils::PhysicalDevice* device{nullptr};

    utils::Buffers uniformBuffersHost;
    utils::Buffers uniformBuffersDevice;

    void createUniformBuffers(uint32_t imageCount);
    BaseCamera& updateViewMatrix();

public:
    BaseCamera();
    BaseCamera(float angle, float aspect, float near);
    BaseCamera(float angle, float aspect, float near, float far);
    void recreate(float angle, float aspect, float near, float far);
    void recreate(float angle, float aspect, float near);

    BaseCamera& setGlobalTransform(const math::Matrix<float,4,4> & transform) override;
    BaseCamera& translate(const math::Vector<float,3> & translate) override;
    BaseCamera& rotate(const float & ang ,const math::Vector<float,3> & ax) override;
    BaseCamera& scale(const math::Vector<float,3> & scale) override;

    BaseCamera& rotate(const math::Quaternion<float>& quat);
    BaseCamera& rotateX(const float & ang ,const math::Vector<float,3> & ax);
    BaseCamera& rotateY(const float & ang ,const math::Vector<float,3> & ax);

    BaseCamera& setProjMatrix(const math::Matrix<float,4,4> & proj);
    BaseCamera& setTranslation(const math::Vector<float,3> & translate);
    BaseCamera& setRotation(const float & ang ,const math::Vector<float,3> & ax);
    BaseCamera& setRotation(const math::Quaternion<float>& rotation);

    void create(const utils::PhysicalDevice& device, uint32_t imageCount) override;
    void update(uint32_t frameNumber, VkCommandBuffer commandBuffer) override;

    const utils::Buffers& getBuffers() const override;

    math::Vector<float,3>         getTranslation()const;
    math::Quaternion<float>       getRotationX()const;
    math::Quaternion<float>       getRotationY()const;

    math::Matrix<float,4,4>       getProjMatrix() const;
    math::Matrix<float,4,4>       getViewMatrix() const;
};

}
#endif // BASECAMERA_H
