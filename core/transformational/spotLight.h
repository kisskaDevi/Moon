#ifndef SPOTLIGHT_H
#define SPOTLIGHT_H

#include "transformational.h"
#include "light.h"
#include "quaternion.h"
#include "buffer.h"

#include <filesystem>

namespace moon::utils { class Texture;}

namespace moon::transformational {

struct LightBufferObject
{
    alignas(16) moon::math::Matrix<float,4,4>   proj;
    alignas(16) moon::math::Matrix<float,4,4>   view;
    alignas(16) moon::math::Matrix<float,4,4>   projView;
    alignas(16) moon::math::Vector<float,4>     position;
    alignas(16) moon::math::Vector<float,4>     lightColor;
    alignas(16) moon::math::Vector<float,4>     lightProp;
};

enum SpotType
{
    circle,
    square
};

class SpotLight : public Transformational, public moon::interfaces::Light
{
private:
    moon::utils::Texture*               tex{nullptr};
    moon::utils::Texture*               emptyTextureBlack{nullptr};
    moon::utils::Texture*               emptyTextureWhite{nullptr};

    float                               lightPowerFactor{10.0f};
    float                               lightDropFactor{1.0f};
    moon::math::Vector<float,4>                     lightColor{0.0f};

    moon::math::Matrix<float,4,4>       projectionMatrix{1.0f};
    bool                                created{false};
    VkDevice                            device{VK_NULL_HANDLE};
    SpotType                            type{SpotType::circle};

    moon::math::Quaternion<float>       translation{0.0f,0.0f,0.0f,0.0f};
    moon::math::Quaternion<float>       rotation{1.0f,0.0f,0.0f,0.0f};
    moon::math::Quaternion<float>       rotationX{1.0f,0.0f,0.0f,0.0f};
    moon::math::Quaternion<float>       rotationY{1.0f,0.0f,0.0f,0.0f};
    moon::math::Vector<float,3>         scaling{1.0f,1.0f,1.0f};
    moon::math::Matrix<float,4,4>       globalTransformation{1.0f};
    moon::math::Matrix<float,4,4>       modelMatrix{1.0f};

    VkDescriptorSetLayout               textureDescriptorSetLayout{VK_NULL_HANDLE};
    std::vector<VkDescriptorSet>        textureDescriptorSets;

    std::vector<moon::utils::Buffer> uniformBuffersHost;
    std::vector<moon::utils::Buffer> uniformBuffersDevice;

    void createUniformBuffers(VkPhysicalDevice physicalDevice, VkDevice device, uint32_t imageCount);
    void createDescriptorPool(VkDevice device, uint32_t imageCount);
    void createDescriptorSets(VkDevice device, uint32_t imageCount);
    void updateDescriptorSets(VkDevice device, uint32_t imageCount);

    void updateUniformBuffersFlags(std::vector<moon::utils::Buffer>& uniformBuffers);
    void updateModelMatrix();
public:
    SpotLight(const moon::math::Vector<float,4>& color, const moon::math::Matrix<float,4,4> & projection, bool enableShadow = true, bool enableScattering = false, SpotType type = SpotType::circle);
    SpotLight(const std::filesystem::path & TEXTURE_PATH, const moon::math::Matrix<float,4,4> & projection, bool enableShadow = true, bool enableScattering = false, SpotType type = SpotType::circle);
    virtual ~SpotLight();

    SpotLight&          setGlobalTransform(const moon::math::Matrix<float,4,4> & transform) override;
    SpotLight&          translate(const moon::math::Vector<float,3> & translate) override;
    SpotLight&          rotate(const float & ang,const moon::math::Vector<float,3> & ax) override;
    SpotLight&          scale(const moon::math::Vector<float,3> & scale) override;

    SpotLight&          rotateX(const float & ang ,const moon::math::Vector<float,3> & ax);
    SpotLight&          rotateY(const float & ang ,const moon::math::Vector<float,3> & ax);
    SpotLight&          setTranslation(const moon::math::Vector<float,3>& translate);
    SpotLight&          setRotation(const moon::math::Quaternion<float>& rotation);
    SpotLight&          setRotation(const float & ang ,const moon::math::Vector<float,3> & ax);
    SpotLight&          rotate(const moon::math::Quaternion<float>& quat);

    void                setLightColor(const moon::math::Vector<float,4> & color);
    void                setLightDropFactor(const float& dropFactor);
    void                setTexture(moon::utils::Texture* tex);
    void                setProjectionMatrix(const moon::math::Matrix<float,4,4> & projection);

    moon::math::Matrix<float,4,4>   getModelMatrix() const;
    moon::math::Vector<float,3>     getTranslate() const;
    moon::math::Vector<float,4>     getLightColor() const;

    void destroy(VkDevice device) override;

    void create(
            moon::utils::PhysicalDevice device,
            VkCommandPool commandPool,
            uint32_t imageCount) override;

    void render(
        uint32_t frameNumber,
        VkCommandBuffer commandBuffer,
        const std::vector<VkDescriptorSet>& descriptorSet,
        VkPipelineLayout pipelineLayout,
        VkPipeline pipeline) override;

    void update(
        uint32_t frameNumber,
        VkCommandBuffer commandBuffer) override;

    void printStatus() const;
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

    void updateModelMatrix();
public:
    IsotropicLight(const moon::math::Vector<float,4>& color, float radius = 100.0f);
    ~IsotropicLight();

    void setLightColor(const moon::math::Vector<float,4> & color);
    void setLightDropFactor(const float& dropFactor);
    void setProjectionMatrix(const moon::math::Matrix<float,4,4> & projection);
    void setTranslation(const moon::math::Vector<float,3>& translate);

    IsotropicLight& setGlobalTransform(const moon::math::Matrix<float,4,4>& transform);
    IsotropicLight& translate(const moon::math::Vector<float,3>& translate);
    IsotropicLight& rotate(const float& ang,const moon::math::Vector<float,3>& ax);
    IsotropicLight& scale(const moon::math::Vector<float,3>& scale);

    void rotateX(const float& ang ,const moon::math::Vector<float,3>& ax);
    void rotateY(const float& ang ,const moon::math::Vector<float,3>& ax);

    moon::math::Vector<float,3> getTranslate() const;
    moon::math::Vector<float,4> getLightColor() const;
    std::vector<SpotLight*> get() const;
};

}
#endif // SPOTLIGHT_H
