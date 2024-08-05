#ifndef SPOTLIGHT_H
#define SPOTLIGHT_H

#include "transformational.h"
#include "light.h"
#include "quaternion.h"
#include "buffer.h"
#include "texture.h"

#include <filesystem>

namespace moon::transformational {

struct LightBufferObject {
    alignas(16) moon::math::Matrix<float,4,4>   proj;
    alignas(16) moon::math::Matrix<float,4,4>   view;
    alignas(16) moon::math::Vector<float,4>     lightColor;
    alignas(16) moon::math::Vector<float,4>     lightProp;
};

class SpotLight : public Transformational, public moon::interfaces::Light
{
public:
    enum Type {
        circle,
        square
    };

private:
    const std::filesystem::path texturePath;
    moon::utils::Texture texture;

    Type                                type{ Type::circle };
    float                               lightPowerFactor{10.0f};
    float                               lightDropFactor{1.0f};
    moon::math::Vector<float,4>         lightColor{0.0f};
    moon::math::Matrix<float,4,4>       projectionMatrix{1.0f};

    moon::math::Quaternion<float>       translation{0.0f,0.0f,0.0f,0.0f};
    moon::math::Quaternion<float>       rotation{1.0f,0.0f,0.0f,0.0f};
    moon::math::Quaternion<float>       rotationX{1.0f,0.0f,0.0f,0.0f};
    moon::math::Quaternion<float>       rotationY{1.0f,0.0f,0.0f,0.0f};
    moon::math::Vector<float,3>         scaling{1.0f,1.0f,1.0f};
    moon::math::Matrix<float,4,4>       globalTransformation{1.0f};
    moon::math::Matrix<float,4,4>       modelMatrix{1.0f};

    moon::utils::vkDefault::DescriptorSetLayout textureDescriptorSetLayout;
    utils::vkDefault::DescriptorSets textureDescriptorSets;

    std::vector<moon::utils::Buffer> uniformBuffersHost;
    std::vector<moon::utils::Buffer> uniformBuffersDevice;

    const moon::utils::PhysicalDevice* device{ nullptr };

    void createUniformBuffers(uint32_t imageCount);
    void createDescriptorPool(uint32_t imageCount);
    void updateDescriptorSets(uint32_t imageCount);

    SpotLight& update() override;

public:

    SpotLight(const moon::math::Vector<float,4>& color, const moon::math::Matrix<float,4,4> & projection, bool enableShadow = true, bool enableScattering = false, Type type = Type::circle);
    SpotLight(const std::filesystem::path& texturePath, const moon::math::Matrix<float,4,4> & projection, bool enableShadow = true, bool enableScattering = false, Type type = Type::circle);
    virtual ~SpotLight();

    SpotLight& setGlobalTransform(const moon::math::Matrix<float,4,4> & transform) override;
    SpotLight& translate(const moon::math::Vector<float,3> & translate) override;
    SpotLight& rotate(const float & ang,const moon::math::Vector<float,3> & ax) override;
    SpotLight& scale(const moon::math::Vector<float,3> & scale) override;

    SpotLight& rotateX(const float & ang ,const moon::math::Vector<float,3> & ax);
    SpotLight& rotateY(const float & ang ,const moon::math::Vector<float,3> & ax);
    SpotLight& setTranslation(const moon::math::Vector<float,3>& translate);
    SpotLight& setRotation(const moon::math::Quaternion<float>& rotation);
    SpotLight& setRotation(const float & ang ,const moon::math::Vector<float,3> & ax);
    SpotLight& rotate(const moon::math::Quaternion<float>& quat) override;

    void setLightColor(const moon::math::Vector<float,4> & color);
    void setLightDropFactor(const float& dropFactor);
    void setProjectionMatrix(const moon::math::Matrix<float,4,4> & projection);

    moon::math::Matrix<float,4,4>   getModelMatrix() const;
    moon::math::Vector<float,3>     getTranslate() const;
    moon::math::Vector<float,4>     getLightColor() const;

    void create(
            const moon::utils::PhysicalDevice& device,
            VkCommandPool commandPool,
            uint32_t imageCount) override;

    void render(
        uint32_t frameNumber,
        VkCommandBuffer commandBuffer,
        const utils::vkDefault::DescriptorSets& descriptorSet,
        VkPipelineLayout pipelineLayout,
        VkPipeline pipeline) override;

    void update(
        uint32_t frameNumber,
        VkCommandBuffer commandBuffer) override;
};

class IsotropicLight: public Transformational
{
private:
    moon::math::Vector<float,4>         lightColor{0.0f};
    moon::math::Matrix<float,4,4>       projectionMatrix{1.0f};
    float                               lightDropFactor{1.0f};

    moon::math::Quaternion<float>       translation{0.0f,0.0f,0.0f,0.0f};
    moon::math::Quaternion<float>       rotation{1.0f,0.0f,0.0f,0.0f};
    moon::math::Quaternion<float>       rotationX{1.0f,0.0f,0.0f,0.0f};
    moon::math::Quaternion<float>       rotationY{1.0f,0.0f,0.0f,0.0f};
    moon::math::Vector<float,3>         scaling{1.0f,1.0f,1.0f};
    moon::math::Matrix<float,4,4>       globalTransformation{1.0f};
    moon::math::Matrix<float,4,4>       modelMatrix{1.0f};

    std::vector<SpotLight*>             lightSource;

    IsotropicLight& update() override;

public:
    IsotropicLight(const moon::math::Vector<float,4>& color, float radius = 100.0f);
    ~IsotropicLight();

    void setLightColor(const moon::math::Vector<float,4> & color);
    void setLightDropFactor(const float& dropFactor);
    void setProjectionMatrix(const moon::math::Matrix<float,4,4> & projection);

    IsotropicLight& setTranslation(const moon::math::Vector<float,3>& translate);
    IsotropicLight& setGlobalTransform(const moon::math::Matrix<float,4,4>& transform) override;
    IsotropicLight& translate(const moon::math::Vector<float,3>& translate) override;
    IsotropicLight& rotate(const float& ang,const moon::math::Vector<float,3>& ax) override;
    IsotropicLight& scale(const moon::math::Vector<float,3>& scale) override;
    IsotropicLight& rotate(const math::Quaternion<float>&) override {return *this;}

    IsotropicLight& rotateX(const float& ang ,const moon::math::Vector<float,3>& ax);
    IsotropicLight& rotateY(const float& ang ,const moon::math::Vector<float,3>& ax);

    moon::math::Vector<float,3> getTranslate() const;
    moon::math::Vector<float,4> getLightColor() const;
    std::vector<SpotLight*> get() const;
};

}
#endif // SPOTLIGHT_H
