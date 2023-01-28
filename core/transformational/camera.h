#ifndef CAMERA_H
#define CAMERA_H

#include <libs/vulkan/vulkan.h>
#include "transformational.h"
#include "libs/quaternion.h"

struct UniformBufferObject{
    alignas(16) glm::mat4           view;
    alignas(16) glm::mat4           proj;
    alignas(16) glm::vec4           eyePosition;
};

class camera : public transformational
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
    };
    std::vector<buffer>     uniformBuffers;
    std::vector<buffer>     uniformBuffersDevice;

    void updateViewMatrix();
public:
    camera();
    ~camera();

    void setGlobalTransform(const glm::mat4 & transform);
    void translate(const glm::vec3 & translate);
    void rotate(const float & ang ,const glm::vec3 & ax);
    void scale(const glm::vec3 & scale);

    void rotateX(const float & ang ,const glm::vec3 & ax);
    void rotateY(const float & ang ,const glm::vec3 & ax);

    void                    setProjMatrix(const glm::mat4 & proj);
    void                    setPosition(const glm::vec3 & translate);
    void                    setRotation(const float & ang ,const glm::vec3 & ax);
    void                    setRotation(const quaternion<float>& rotation);
    void                    setRotations(const quaternion<float>& rotationX, const quaternion<float>& rotationY);

    void destroyUniformBuffers(VkDevice* device);
    void createUniformBuffers(VkPhysicalDevice* physicalDevice, VkDevice* device, uint32_t imageCount);
    void updateUniformBuffer(VkDevice device, VkCommandBuffer commandBuffer, uint32_t frameNumber);

    VkBuffer                getBuffer(uint32_t index)const;
    glm::vec3               getTranslation()const;
    quaternion<float>       getRotationX()const;
    quaternion<float>       getRotationY()const;

    glm::mat4x4             getProjMatrix() const;
    glm::mat4x4             getViewMatrix() const;
};

#endif // CAMERA_H
