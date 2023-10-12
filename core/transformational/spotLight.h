#ifndef SPOTLIGHT_H
#define SPOTLIGHT_H

#include "transformational.h"
#include "light.h"
#include "quaternion.h"
#include "buffer.h"

#include <filesystem>

class shadowGraphics;

struct LightBufferObject
{
    alignas(16) matrix<float,4,4>   proj;
    alignas(16) matrix<float,4,4>   view;
    alignas(16) matrix<float,4,4>   projView;
    alignas(16) vector<float,4>     position;
    alignas(16) vector<float,4>     lightColor;
    alignas(16) vector<float,4>     lightProp;
};

enum spotType
{
    circle,
    square
};

class spotLight : public transformational, public light
{
private:
    attachments*                        shadow{nullptr};
    texture*                            tex{nullptr};
    VkExtent2D                          shadowExtent{1024,1024};

    float                               lightPowerFactor{10.0f};
    float                               lightDropFactor{1.0f};
    vector<float,4>                     lightColor{0.0f};

    matrix<float,4,4>                   projectionMatrix{1.0f};
    bool                                enableShadow{false};
    bool                                enableScattering{false};
    bool                                created{false};
    uint32_t                            type{spotType::circle};

    quaternion<float>                   translation{0.0f,0.0f,0.0f,0.0f};
    quaternion<float>                   rotation{1.0f,0.0f,0.0f,0.0f};
    quaternion<float>                   rotationX{1.0f,0.0f,0.0f,0.0f};
    quaternion<float>                   rotationY{1.0f,0.0f,0.0f,0.0f};
    vector<float,3>                     scaling{1.0f,1.0f,1.0f};
    matrix<float,4,4>                   globalTransformation{1.0f};
    matrix<float,4,4>                   modelMatrix{1.0f};

    VkDescriptorSetLayout               bufferDescriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorSetLayout               descriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorPool                    descriptorPool{VK_NULL_HANDLE};
    std::vector<VkDescriptorSet>        bufferDescriptorSets;
    std::vector<VkDescriptorSet>        descriptorSets;

    std::vector<buffer> uniformBuffersHost;
    std::vector<buffer> uniformBuffersDevice;

    void updateUniformBuffersFlags(std::vector<buffer>& uniformBuffers);
    void updateModelMatrix();
public:
    spotLight(const matrix<float,4,4> & projection, bool enableShadow = true, bool enableScattering = false, uint32_t type = spotType::circle);
    spotLight(const std::filesystem::path & TEXTURE_PATH, const matrix<float,4,4> & projection, bool enableShadow = true, bool enableScattering = false, uint32_t type = spotType::circle);
    ~spotLight();
    void destroy(VkDevice device) override;

    spotLight&          setGlobalTransform(const matrix<float,4,4> & transform) override;
    spotLight&          translate(const vector<float,3> & translate) override;
    spotLight&          rotate(const float & ang,const vector<float,3> & ax) override;
    spotLight&          scale(const vector<float,3> & scale) override;
    spotLight&          rotateX(const float & ang ,const vector<float,3> & ax);
    spotLight&          rotateY(const float & ang ,const vector<float,3> & ax);
    spotLight&          setTranslation(const vector<float,3>& translate);
    spotLight&          setRotation(const quaternion<float>& rotation);
    spotLight&          setRotation(const float & ang ,const vector<float,3> & ax);
    spotLight&          rotate(const quaternion<float>& quat);

    void                setLightColor(const vector<float,4> & color);
    void                setLightDropFactor(const float& dropFactor);
    void                setShadowExtent(const VkExtent2D & shadowExtent);
    void                setShadow(bool enable);
    void                setScattering(bool enable);
    void                setTexture(texture* tex);
    void                setProjectionMatrix(const matrix<float,4,4> & projection);

    matrix<float,4,4>   getModelMatrix() const;
    vector<float,3>     getTranslate() const;
    vector<float,4>     getLightColor() const;

    texture*            getTexture() override;
    attachments*        getAttachments() override;
    uint8_t             getPipelineBitMask() const override;

    bool                isShadowEnable() const override;
    bool                isScatteringEnable() const override;

    VkDescriptorSet*    getDescriptorSets() override;
    VkDescriptorSet*    getBufferDescriptorSets() override;

    void create(
            physicalDevice device,
            VkCommandPool commandPool,
            uint32_t imageCount,
            texture* emptyTextureBlack = nullptr,
            texture* emptyTextureWhite = nullptr) override;

    void                updateUniformBuffer(VkCommandBuffer commandBuffer, uint32_t frameNumber) override;

    void                createUniformBuffers(VkPhysicalDevice physicalDevice, VkDevice device, uint32_t imageCount);
    void                createDescriptorPool(VkDevice device, uint32_t imageCount);
    void                createDescriptorSets(VkDevice device, uint32_t imageCount);
    void                updateDescriptorSets(VkDevice device, uint32_t imageCount, texture* emptyTextureBlack , texture* emptyTextureWhite);
};

class isotropicLight: public transformational
{
private:
    vector<float,4>                     lightColor{0.0f};
    matrix<float,4,4>                   projectionMatrix{1.0f};
    float                               lightDropFactor{1.0f};

    quaternion<float>                   translation{0.0f,0.0f,0.0f,0.0f};
    quaternion<float>                   rotation{1.0f,0.0f,0.0f,0.0f};
    quaternion<float>                   rotationX{1.0f,0.0f,0.0f,0.0f};
    quaternion<float>                   rotationY{1.0f,0.0f,0.0f,0.0f};
    vector<float,3>                     scaling{1.0f,1.0f,1.0f};
    matrix<float,4,4>                   globalTransformation{1.0f};
    matrix<float,4,4>                   modelMatrix{1.0f};

    std::vector<spotLight *> lightSource;

    void updateModelMatrix();
public:
    isotropicLight(std::vector<spotLight *>& lightSource, float radius = 100.0f);
    ~isotropicLight();

    void setLightColor(const vector<float,4> & color);
    void setLightDropFactor(const float& dropFactor);
    void setProjectionMatrix(const matrix<float,4,4> & projection);
    void setTranslation(const vector<float,3>& translate);

    isotropicLight& setGlobalTransform(const matrix<float,4,4>& transform);
    isotropicLight& translate(const vector<float,3>& translate);
    isotropicLight& rotate(const float& ang,const vector<float,3>& ax);
    isotropicLight& scale(const vector<float,3>& scale);

    void rotateX(const float& ang ,const vector<float,3>& ax);
    void rotateY(const float& ang ,const vector<float,3>& ax);

    vector<float,3> getTranslate() const;
    vector<float,4> getLightColor() const;
};

#endif // SPOTLIGHT_H
