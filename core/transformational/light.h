#ifndef LIGHT_H
#define LIGHT_H

#include "transformational.h"
#include "lightInterface.h"
#include "libs/quaternion.h"

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
    attachments*                        shadow{nullptr};
    texture*                            tex{nullptr};
    VkExtent2D                          shadowExtent{1024,1024};

    bool                                enableShadow{false};
    bool                                enableScattering{false};

    uint32_t                            type{spotType::circle};
    float                               lightPowerFactor{10.0f};
    float                               lightDropFactor{1.0f};
    glm::vec4                           lightColor{0.0f};
    glm::mat4x4                         projectionMatrix{1.0f};

    quaternion<float>                   translation{0.0f,0.0f,0.0f,0.0f};
    quaternion<float>                   rotation{1.0f,0.0f,0.0f,0.0f};
    quaternion<float>                   rotationX{1.0f,0.0f,0.0f,0.0f};
    quaternion<float>                   rotationY{1.0f,0.0f,0.0f,0.0f};
    glm::vec3                           scaling{1.0f,1.0f,1.0f};
    glm::mat4x4                         globalTransformation{1.0f};
    glm::mat4x4                         modelMatrix{1.0f};

    VkDescriptorSetLayout               descriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorSetLayout               shadowDescriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorPool                    descriptorPool{VK_NULL_HANDLE};
    std::vector<VkDescriptorSet>        descriptorSets;
    std::vector<VkDescriptorSet>        shadowDescriptorSets;

    struct buffer{
        VkBuffer       instance{VK_NULL_HANDLE};
        VkDeviceMemory memory{VK_NULL_HANDLE};
        bool           updateFlag{true};
        void*          map{nullptr};
    };
    std::vector<buffer> uniformBuffersHost;
    std::vector<buffer> uniformBuffersDevice;

    void updateUniformBuffersFlags(std::vector<buffer>& uniformBuffers);
    void destroyUniformBuffers(VkDevice* device, std::vector<buffer>& uniformBuffers);
    void updateModelMatrix();
public:
    spotLight(bool enableShadow = true, bool enableScattering = false, uint32_t type = spotType::circle);
    spotLight(const std::string & TEXTURE_PATH, bool enableShadow = true, bool enableScattering = false, uint32_t type = spotType::circle);
    ~spotLight();
    void destroy(VkDevice* device) override;

    void                setGlobalTransform(const glm::mat4 & transform) override;
    void                translate(const glm::vec3 & translate) override;
    void                rotate(const float & ang,const glm::vec3 & ax) override;
    void                scale(const glm::vec3 & scale) override;
    void                rotateX(const float & ang ,const glm::vec3 & ax);
    void                rotateY(const float & ang ,const glm::vec3 & ax);
    void                setPosition(const glm::vec3& translate);

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
    attachments*        getAttachments() override;
    uint8_t             getPipelineBitMask() override;

    bool                isShadowEnable() const override;
    bool                isScatteringEnable() const;

    VkDescriptorSet*    getDescriptorSets() override;
    VkDescriptorSet*    getShadowDescriptorSets() override;

    void                createUniformBuffers(VkPhysicalDevice* physicalDevice, VkDevice* device, uint32_t imageCount) override;
    void                updateUniformBuffer(VkCommandBuffer commandBuffer, uint32_t frameNumber) override;

    void                createDescriptorPool(VkDevice* device, uint32_t imageCount) override;
    void                createDescriptorSets(VkDevice* device, uint32_t imageCount) override;
    void                updateDescriptorSets(VkDevice* device, uint32_t imageCount, texture* emptyTexture) override;
};

class isotropicLight: public transformational
{
private:
    glm::vec4                           lightColor{0.0f};
    glm::mat4x4                         projectionMatrix{1.0f};

    quaternion<float>                   translation{0.0f,0.0f,0.0f,0.0f};
    quaternion<float>                   rotation{1.0f,0.0f,0.0f,0.0f};
    quaternion<float>                   rotationX{1.0f,0.0f,0.0f,0.0f};
    quaternion<float>                   rotationY{1.0f,0.0f,0.0f,0.0f};
    glm::vec3                           scaling{1.0f,1.0f,1.0f};
    glm::mat4x4                         globalTransformation{1.0f};
    glm::mat4x4                         modelMatrix{1.0f};

    std::vector<spotLight *> lightSource;

    void updateModelMatrix();
public:
    isotropicLight(std::vector<spotLight *>& lightSource);
    ~isotropicLight();

    void setLightColor(const glm::vec4 & color);
    void setProjectionMatrix(const glm::mat4x4 & projection);
    void setPosition(const glm::vec3& translate);

    void setGlobalTransform(const glm::mat4& transform);
    void translate(const glm::vec3& translate);
    void rotate(const float& ang,const glm::vec3& ax);
    void scale(const glm::vec3& scale);

    void rotateX(const float& ang ,const glm::vec3& ax);
    void rotateY(const float& ang ,const glm::vec3& ax);

    glm::vec3 getTranslate() const;
    glm::vec4 getLightColor() const;
};

#endif // LIGHT_H
