#ifndef LIGHT_H
#define LIGHT_H

#include "transformational.h"
#include "core/graphics/graphics.h"

class texture;
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

class spotLight : public transformational
{
private:
    shadowGraphics                      *shadow = nullptr;
    texture                             *tex = nullptr;
    VkExtent2D                          shadowExtent = {1024,1024};

    bool                                enableShadow = false;
    bool                                enableScattering = false;

    uint32_t                            type;
    float                               lightDropFactor;
    float                               lightPowerFactor;
    glm::vec4                           lightColor;

    glm::mat4x4                         projectionMatrix;
    glm::mat4x4                         viewMatrix;
    glm::mat4x4                         modelMatrix;

    glm::mat4x4                         m_globalTransform;
    glm::vec3                           m_translate;
    glm::vec3                           m_scale;
    glm::quat                           m_rotate;
    glm::quat                           m_rotateX;
    glm::quat                           m_rotateY;

    VkDescriptorPool                    descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet>        descriptorSets;
    std::vector<VkBuffer>               uniformBuffers;
    std::vector<VkDeviceMemory>         uniformBuffersMemory;

    void updateViewMatrix();
public:
    spotLight(uint32_t type = spotType::circle);
    spotLight(const std::string & TEXTURE_PATH, uint32_t type = spotType::circle);
    ~spotLight();
    void cleanup(VkDevice* device);
    void destroyBuffer(VkDevice* device);

    void setGlobalTransform(const glm::mat4 & transform);
    void translate(const glm::vec3 & translate);
    void rotate(const float & ang,const glm::vec3 & ax);
    void scale(const glm::vec3 & scale);
    void setPosition(const glm::vec3& translate);

    void rotateX(const float & ang ,const glm::vec3 & ax);
    void rotateY(const float & ang ,const glm::vec3 & ax);

    void setProjectionMatrix(const glm::mat4x4 & projection);
    void createUniformBuffers(VkPhysicalDevice* physicalDevice, VkDevice* device, uint32_t imageCount);
    void createShadow(VkPhysicalDevice* physicalDevice, VkDevice* device, QueueFamilyIndices* queueFamilyIndices, uint32_t imageCount);

    void                            updateLightBuffer(VkDevice* device, uint32_t frameNumber);

    void                            setLightColor(const glm::vec4 & color);
    void                            setShadowExtent(const VkExtent2D & shadowExtent);
    void                            setScattering(bool enable);
    void                            setTexture(texture* tex);

    glm::mat4x4                     getViewMatrix() const;
    glm::mat4x4                     getModelMatrix() const;
    glm::vec3                       getTranslate() const;
    glm::vec4                       getLightColor() const;
    uint32_t                        getLightNumber() const;
    texture*                        getTexture();

    VkDescriptorPool&               getDescriptorPool();
    std::vector<VkDescriptorSet>&   getDescriptorSets();
    std::vector<VkBuffer>&          getUniformBuffers();

    bool                            isShadowEnable() const;
    bool                            isScatteringEnable() const;

    void                            updateShadowCommandBuffer(uint32_t frameNumber, std::vector<object*>& objects);
    void                            createShadowCommandBuffers();
    void                            updateShadowDescriptorSets();
    std::vector<VkCommandBuffer>&   getShadowCommandBuffer();
    VkImageView&                    getShadowImageView();
    VkSampler&                      getShadowSampler();
};

class pointLight : public transformational
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
    pointLight(std::vector<spotLight *>& lightSource);
    ~pointLight();

    void setLightColor(const glm::vec4 & color);

    void setGlobalTransform(const glm::mat4& transform);
    void translate(const glm::vec3& translate);
    void rotate(const float& ang,const glm::vec3& ax);
    void scale(const glm::vec3& scale);
    void setPosition(const glm::vec3& translate);

    void updateViewMatrix();

    void rotateX(const float& ang ,const glm::vec3& ax);
    void rotateY(const float& ang ,const glm::vec3& ax);

    void setProjectionMatrix(const glm::mat4x4 & projection);

    glm::vec3 getTranslate() const;
    glm::vec4 getLightColor() const;
};

#endif // LIGHT_H
