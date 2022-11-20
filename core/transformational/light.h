#ifndef LIGHT_H
#define LIGHT_H

#include "transformational.h"
#include "lightInterface.h"

class shadowGraphics;

struct LightBufferObject
{
    alignas(16) glm::mat4   proj;
    alignas(16) glm::mat4   view;
    alignas(16) glm::mat4   projView;
    alignas(16) glm::vec4   position;
    alignas(16) glm::vec4   lightColor;
    alignas(16) glm::vec4   lightProp;
};

enum spotType
{
    circle,
    square
};

class spotLight : public transformational, public light
{
private:
    shadowGraphics*                     shadow{nullptr};
    texture*                            tex{nullptr};
    VkExtent2D                          shadowExtent{1024,1024};

    bool                                enableShadow{false};
    bool                                enableScattering{false};

    uint32_t                            type{spotType::circle};
    float                               lightPowerFactor{1.0f};
    float                               lightDropFactor{0.1f};
    glm::vec4                           lightColor{0.0f};

    glm::mat4x4                         projectionMatrix{1.0f};
    glm::mat4x4                         modelMatrix{1.0f};

    glm::mat4x4                         globalTransformation{1.0f};
    glm::vec3                           translation{0.0f};
    glm::vec3                           scaling{1.0f};
    glm::quat                           rotation{1.0f,0.0f,0.0f,0.0f};
    glm::quat                           rotationX{1.0f,0.0f,0.0f,0.0f};
    glm::quat                           rotationY{1.0f,0.0f,0.0f,0.0f};

    VkDescriptorSetLayout               descriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorPool                    descriptorPool{VK_NULL_HANDLE};
    std::vector<VkDescriptorSet>        descriptorSets;
    std::vector<VkBuffer>               uniformBuffers;
    std::vector<VkDeviceMemory>         uniformBuffersMemory;

    void updateModelMatrix();
public:
    spotLight(bool enableShadow = true, bool enableScattering = false, uint32_t type = spotType::circle);
    spotLight(const std::string & TEXTURE_PATH, bool enableShadow = true, bool enableScattering = false, uint32_t type = spotType::circle);
    ~spotLight();
    void destroyUniformBuffers(VkDevice* device) override;
    void destroy(VkDevice* device) override;

    void                setGlobalTransform(const glm::mat4 & transform) override;
    void                translate(const glm::vec3 & translate) override;
    void                rotate(const float & ang,const glm::vec3 & ax) override;
    void                scale(const glm::vec3 & scale) override;
    void                setPosition(const glm::vec3& translate);
    void                rotateX(const float & ang ,const glm::vec3 & ax);
    void                rotateY(const float & ang ,const glm::vec3 & ax);

    void                setLightColor(const glm::vec4 & color);
    void                setShadowExtent(const VkExtent2D & shadowExtent);
    void                setShadow(bool enable);
    void                setScattering(bool enable);
    void                setTexture(texture* tex);
    void                setProjectionMatrix(const glm::mat4x4 & projection);

    glm::mat4x4         getModelMatrix() const;
    glm::vec3           getTranslate() const;
    glm::vec4           getLightColor() const;
    texture*            getTexture() override;

    uint8_t             getPipelineBitMask() override;

    bool                isShadowEnable() const override;
    bool                isScatteringEnable() const;

    VkDescriptorSet*    getDescriptorSets() override;
    VkCommandBuffer*    getShadowCommandBuffer(uint32_t imageCount) override;

    void                createUniformBuffers(VkPhysicalDevice* physicalDevice, VkDevice* device, uint32_t imageCount) override;
    void                updateUniformBuffer(VkDevice* device, uint32_t frameNumber) override;

    void                createShadow(VkPhysicalDevice* physicalDevice, VkDevice* device, QueueFamilyIndices* queueFamilyIndices, uint32_t imageCount, const std::string& ExternalPath) override;
    void                updateShadowDescriptorSets() override;
    void                createShadowCommandBuffers() override;
    void                updateShadowCommandBuffer(uint32_t frameNumber, std::vector<object*>& objects) override;

    void                createDescriptorPool(VkDevice* device, uint32_t imageCount) override;
    void                createDescriptorSets(VkDevice* device, uint32_t imageCount) override;
    void                updateDescriptorSets(VkDevice* device, uint32_t imageCount, texture* emptyTexture) override;
};

class isotropicLight: public transformational
{
private:
    glm::mat4 projectionMatrix;

    glm::vec3 m_translate;
    glm::quat m_rotate;
    glm::vec3 m_scale;
    glm::mat4x4 m_globalTransform;
    glm::quat m_rotateX;
    glm::quat m_rotateY;

    glm::vec4 lightColor;

    std::vector<spotLight *> lightSource;

public:
    isotropicLight(std::vector<spotLight *>& lightSource);
    ~isotropicLight();

    void setLightColor(const glm::vec4 & color);

    void setGlobalTransform(const glm::mat4& transform);
    void translate(const glm::vec3& translate);
    void rotate(const float& ang,const glm::vec3& ax);
    void scale(const glm::vec3& scale);
    void setPosition(const glm::vec3& translate);
    void setProjectionMatrix(const glm::mat4x4 & projection);

    void updateViewMatrix();

    void rotateX(const float& ang ,const glm::vec3& ax);
    void rotateY(const float& ang ,const glm::vec3& ax);

    glm::vec3 getTranslate() const;
    glm::vec4 getLightColor() const;
};

#endif // LIGHT_H
