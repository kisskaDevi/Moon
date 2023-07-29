#ifndef BASECAMERA_H
#define BASECAMERA_H

#include <vector>
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

class baseCamera : public transformational, public camera
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
    std::vector<buffer>     uniformBuffersHost;
    std::vector<buffer>     uniformBuffersDevice;

    void updateUniformBuffersFlags(std::vector<buffer>& uniformBuffers);
    void destroyUniformBuffers(VkDevice device, std::vector<buffer>& uniformBuffers);
    void updateViewMatrix();
public:
    baseCamera();
    baseCamera(float angle, float aspect, float near, float far);
    ~baseCamera();
    void destroy(VkDevice device) override;
    void recreate(float angle, float aspect, float near, float far);

    void setGlobalTransform(const matrix<float,4,4> & transform) override;
    void translate(const vector<float,3> & translate) override;
    void rotate(const float & ang ,const vector<float,3> & ax) override;
    void scale(const vector<float,3> & scale) override;

    void rotate(const quaternion<float>& quat);
    void rotateX(const float & ang ,const vector<float,3> & ax);
    void rotateY(const float & ang ,const vector<float,3> & ax);

    void                    setProjMatrix(const matrix<float,4,4> & proj);
    void                    setPosition(const vector<float,3> & translate);
    void                    setRotation(const float & ang ,const vector<float,3> & ax);
    void                    setRotation(const quaternion<float>& rotation);
    void                    setRotations(const quaternion<float>& rotationX, const quaternion<float>& rotationY);

    void createUniformBuffers(VkPhysicalDevice physicalDevice, VkDevice device, uint32_t imageCount) override;
    void updateUniformBuffer(VkCommandBuffer commandBuffer, uint32_t frameNumber) override;

    VkBuffer                getBuffer(uint32_t index) const override;
    VkDeviceSize            getBufferRange() const override;
    vector<float,3>         getTranslation()const;
    quaternion<float>       getRotationX()const;
    quaternion<float>       getRotationY()const;

    matrix<float,4,4>       getProjMatrix() const;
    matrix<float,4,4>       getViewMatrix() const;
};

#endif // BASECAMERA_H
