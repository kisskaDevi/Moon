#ifndef BASEOBJECT_H
#define BASEOBJECT_H

#include <vulkan.h>
#include <filesystem>

#include "transformational.h"
#include "quaternion.h"
#include "texture.h"
#include "buffer.h"
#include "object.h"
#include "matrix.h"

namespace moon::interfaces { class Model;}

namespace moon::transformational {

struct UniformBuffer
{
    alignas(16) moon::math::Matrix<float,4,4> modelMatrix;
    alignas(16) moon::math::Vector<float,4>   constantColor;
    alignas(16) moon::math::Vector<float,4>   colorFactor;
    alignas(16) moon::math::Vector<float,4>   bloomColor;
    alignas(16) moon::math::Vector<float,4>   bloomFactor;
};

class BaseObject : public moon::interfaces::Object, public Transformational
{
private:
    moon::math::Quaternion<float>               translation{0.0f,0.0f,0.0f,0.0f};
    moon::math::Quaternion<float>               rotation{1.0f,0.0f,0.0f,0.0f};
    moon::math::Vector<float,3>                 scaling{1.0f,1.0f,1.0f};
    moon::math::Matrix<float,4,4>               globalTransformation{1.0f};
    moon::math::Matrix<float,4,4>               modelMatrix{1.0f};

    moon::math::Vector<float,4>                 colorFactor{1.0f};
    moon::math::Vector<float,4>                 constantColor{0.0f};
    moon::math::Vector<float,4>                 bloomFactor{1.0f};
    moon::math::Vector<float,4>                 bloomColor{0.0f};

    bool changeAnimationFlag{false};
    uint32_t animationIndex{0};
    uint32_t newAnimationIndex{0};
    float animationTimer{0.0f};
    float startTimer{0.0f};
    float changeAnimationTime{0.0f};

protected:
    const moon::utils::PhysicalDevice* device{nullptr};

    std::vector<moon::utils::Buffer> uniformBuffersHost;
    std::vector<moon::utils::Buffer> uniformBuffersDevice;

    void createUniformBuffers(uint32_t imageCount);

private:
    void createDescriptorPool(uint32_t imageCount);
    void createDescriptorSet(uint32_t imageCount);
    void updateUniformBuffersFlags(std::vector<moon::utils::Buffer>& uniformBuffers);
    void updateModelMatrix();

public:
    BaseObject() = default;
    BaseObject(moon::interfaces::Model* model, uint32_t firstInstance = 0, uint32_t instanceCount = 1);
    virtual ~BaseObject() = default;

    BaseObject& setGlobalTransform(const moon::math::Matrix<float,4,4>& transform) override;
    BaseObject& translate(const moon::math::Vector<float,3>& translate) override;
    BaseObject& rotate(const float& ang, const moon::math::Vector<float,3>& ax) override;
    BaseObject& rotate(const moon::math::Quaternion<float>& quat);
    BaseObject& scale(const moon::math::Vector<float,3>& scale) override;
    BaseObject& setTranslation(const moon::math::Vector<float,3>& translate);

    const moon::math::Vector<float,3> getTranslation() const;
    const moon::math::Quaternion<float> getRotation() const;
    const moon::math::Vector<float,3> getScale() const;

    BaseObject& setConstantColor(const moon::math::Vector<float,4> & color);
    BaseObject& setColorFactor(const moon::math::Vector<float,4> & color);
    BaseObject& setBloomColor(const moon::math::Vector<float,4> & color);
    BaseObject& setBloomFactor(const moon::math::Vector<float,4> & color);

    void create(
        const moon::utils::PhysicalDevice& device,
        VkCommandPool commandPool,
        uint32_t imageCount) override;

    void update(
        uint32_t frameNumber,
        VkCommandBuffer commandBuffer) override;

    uint32_t getAnimationIndex();
    void setAnimation(uint32_t animationIndex, float animationTime);
    void changeAnimation(uint32_t newAnimationIndex, float changeAnimationTime);
    void updateAnimation(uint32_t imageNumber, float frameTime);

    void printStatus() const;
};

class SkyboxObject : public BaseObject{
private:
    moon::utils::CubeTexture* texture{nullptr};

    void createDescriptorPool(uint32_t imageCount);
    void createDescriptorSet(uint32_t imageCount);
public:
    SkyboxObject(const std::vector<std::filesystem::path>& TEXTURE_PATH);
    ~SkyboxObject();

    SkyboxObject& setMipLevel(float mipLevel);
    SkyboxObject& translate(const moon::math::Vector<float,3>& translate) override;

    void create(
        const moon::utils::PhysicalDevice& device,
        VkCommandPool commandPool,
        uint32_t imageCount) override;
};

}
#endif // BASEOBJECT_H
