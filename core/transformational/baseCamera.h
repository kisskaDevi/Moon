#ifndef BASECAMERA_H
#define BASECAMERA_H

#include <vulkan.h>

#include "camera.h"
#include "transformational.h"
#include "quaternion.h"
#include "buffer.h"

struct UniformBufferObject{
    alignas(16) matrix<float,4,4>   view;
    alignas(16) matrix<float,4,4>   proj;
    alignas(16) vector<float,4>     eyePosition;
};

class baseCamera : public transformational, public moon::interfaces::Camera
{
private:
    matrix<float,4,4>       projMatrix{1.0f};
    matrix<float,4,4>       viewMatrix{1.0f};
    matrix<float,4,4>       globalTransformation{1.0f};

    quaternion<float>       translation{0.0f,0.0f,0.0f,0.0f};
    quaternion<float>       rotation{1.0f,0.0f,0.0f,0.0f};
    quaternion<float>       rotationX{1.0f,0.0f,0.0f,0.0f};
    quaternion<float>       rotationY{1.0f,0.0f,0.0f,0.0f};

protected:
    bool                    created{false};
    VkDevice                device{VK_NULL_HANDLE};

    moon::utils::Buffers    uniformBuffersHost;
    moon::utils::Buffers    uniformBuffersDevice;

    void createUniformBuffers(VkPhysicalDevice physicalDevice, VkDevice device, uint32_t imageCount);
    void updateViewMatrix();
public:
    baseCamera();
    baseCamera(float angle, float aspect, float near);
    baseCamera(float angle, float aspect, float near, float far);
    ~baseCamera();
    void destroy(VkDevice device) override;
    void recreate(float angle, float aspect, float near, float far);
    void recreate(float angle, float aspect, float near);

    baseCamera& setGlobalTransform(const matrix<float,4,4> & transform) override;
    baseCamera& translate(const vector<float,3> & translate) override;
    baseCamera& rotate(const float & ang ,const vector<float,3> & ax) override;
    baseCamera& scale(const vector<float,3> & scale) override;

    baseCamera& rotate(const quaternion<float>& quat);
    baseCamera& rotateX(const float & ang ,const vector<float,3> & ax);
    baseCamera& rotateY(const float & ang ,const vector<float,3> & ax);

    baseCamera& setProjMatrix(const matrix<float,4,4> & proj);
    baseCamera& setTranslation(const vector<float,3> & translate);
    baseCamera& setRotation(const float & ang ,const vector<float,3> & ax);
    baseCamera& setRotation(const quaternion<float>& rotation);

    void create(moon::utils::PhysicalDevice device, uint32_t imageCount) override;
    void update(uint32_t frameNumber, VkCommandBuffer commandBuffer) override;

    const moon::utils::Buffers& getBuffers() const override;

    vector<float,3>         getTranslation()const;
    quaternion<float>       getRotationX()const;
    quaternion<float>       getRotationY()const;

    matrix<float,4,4>       getProjMatrix() const;
    matrix<float,4,4>       getViewMatrix() const;
};

#endif // BASECAMERA_H
