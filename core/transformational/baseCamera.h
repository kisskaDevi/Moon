#ifndef BASECAMERA_H
#define BASECAMERA_H

#include <vulkan.h>
#include "../interfaces/camera.h"
#include "transformational.h"
#include "quaternion.h"

struct UniformBufferObject{
    alignas(16) glm::mat4           view;
    alignas(16) glm::mat4           proj;
    alignas(16) glm::vec4           eyePosition;
};

class baseCamera : public transformational, public camera
{
private:
    glm::mat4x4             projMatrix{1.0f};
    glm::mat4x4             viewMatrix{1.0f};
    glm::mat4x4             globalTransformation{1.0f};

    quaternion<float>       translation{0.0f,0.0f,0.0f,0.0f};
    quaternion<float>       rotation{1.0f,0.0f,0.0f,0.0f};
    quaternion<float>       rotationX{1.0f,0.0f,0.0f,0.0f};
    quaternion<float>       rotationY{1.0f,0.0f,0.0f,0.0f};

protected:
    struct buffer{
        VkBuffer       instance{VK_NULL_HANDLE};
        VkDeviceMemory memory{VK_NULL_HANDLE};
        bool           updateFlag{true};
        void*          map{nullptr};
    };
    std::vector<buffer>     uniformBuffersHost;
    std::vector<buffer>     uniformBuffersDevice;

    void updateUniformBuffersFlags(std::vector<buffer>& uniformBuffers);
    void destroyUniformBuffers(VkDevice* device, std::vector<buffer>& uniformBuffers);
    void updateViewMatrix();
public:
    baseCamera();
    ~baseCamera();
    void destroy(VkDevice* device) override;

    void setGlobalTransform(const glm::mat4 & transform) override;
    void translate(const glm::vec3 & translate) override;
    void rotate(const float & ang ,const glm::vec3 & ax) override;
    void scale(const glm::vec3 & scale) override;

    void rotateX(const float & ang ,const glm::vec3 & ax);
    void rotateY(const float & ang ,const glm::vec3 & ax);

    void                    setProjMatrix(const glm::mat4 & proj);
    void                    setPosition(const glm::vec3 & translate);
    void                    setRotation(const float & ang ,const glm::vec3 & ax);
    void                    setRotation(const quaternion<float>& rotation);
    void                    setRotations(const quaternion<float>& rotationX, const quaternion<float>& rotationY);

    void createUniformBuffers(VkPhysicalDevice* physicalDevice, VkDevice* device, uint32_t imageCount) override;
    void updateUniformBuffer(VkCommandBuffer commandBuffer, uint32_t frameNumber) override;

    VkBuffer                getBuffer(uint32_t index) const override;
    VkDeviceSize            getBufferRange() const override;
    glm::vec3               getTranslation()const;
    quaternion<float>       getRotationX()const;
    quaternion<float>       getRotationY()const;

    glm::mat4x4             getProjMatrix() const;
    glm::mat4x4             getViewMatrix() const;
};

#endif // BASECAMERA_H
