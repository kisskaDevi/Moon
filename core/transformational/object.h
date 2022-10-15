#ifndef OBJECT_H
#define OBJECT_H

#include <libs/vulkan/vulkan.h>
#include "transformational.h"

#include <array>
#include <vector>

struct UniformBuffer
{
    alignas(16) glm::mat4x4 modelMatrix;
    alignas(16) glm::vec4   color;
};

struct gltfModel;
struct Node;
struct Material;

class object : public transformational
{
private:
    gltfModel**                     pModel;
    uint32_t                        modelCount;

    VkDescriptorSetLayout           descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool                descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet>    descriptors;

    glm::mat4x4                     modelMatrix;
    glm::vec3                       m_translate;
    glm::quat                       m_rotate;
    glm::vec3                       m_scale;
    glm::mat4x4                     m_globalTransform;

    float                           visibilityDistance = 10.0f;
    glm::vec4                       color;

    std::vector<VkBuffer>           uniformBuffers;
    std::vector<VkDeviceMemory>     uniformBuffersMemory;
    bool                            enable = true;

    uint32_t                        firstPrimitive;
    uint32_t                        primitiveCount = 0;

    struct Stencil{
        bool                        Enable;
        float                       Width;
        glm::vec4                   Color;
    }stencil;

    void updateModelMatrix();

public:
    object();
    object(uint32_t modelCount, gltfModel** model);
    ~object();
    void destroyUniformBuffers(VkDevice* device);
    void destroy(VkDevice* device);

    VkDescriptorPool&               getDescriptorPool();
    std::vector<VkDescriptorSet>&   getDescriptorSet();
    std::vector<VkBuffer>&          getUniformBuffers();

    void createUniformBuffers(VkPhysicalDevice* physicalDevice, VkDevice* device, uint32_t imageCount);
    void updateUniformBuffer(VkDevice* device, uint32_t currentImage);
    void updateAnimation(uint32_t imageNumber);

    void                setGlobalTransform(const glm::mat4& transform);
    void                translate(const glm::vec3& translate);
    void                rotate(const float& ang, const glm::vec3& ax);
    void                scale(const glm::vec3& scale);
    void                setPosition(const glm::vec3& translate);

    glm::mat4x4&        ModelMatrix();
    glm::mat4x4&        Transformation();
    glm::vec3&          Translate();
    glm::quat&          Rotate();
    glm::vec3&          Scale();

    void                setModel(gltfModel** model3D);
    void                setVisibilityDistance(float visibilityDistance);
    void                setColor(const glm::vec4 & color);

    gltfModel*          getModel(uint32_t index);
    float               getVisibilityDistance() const;
    glm::vec4           getColor() const;

    void                setEnable(const bool& enable);
    bool                getEnable() const;

    void                setStencilEnable(const bool& enable);
    void                setStencilWidth(const float& width);
    void                setStencilColor(const glm::vec4& color);

    bool                getStencilEnable() const;
    float               getStencilWidth() const;
    glm::vec4           getStencilColor() const;

    void                setFirstPrimitive(uint32_t firstPrimitive);
    void                setPrimitiveCount(uint32_t primitiveCount);
    void                resetPrimitiveCount();
    void                increasePrimitiveCount();

    bool                comparePrimitive(uint32_t primitive);
    uint32_t            getFirstPrimitive() const;
    uint32_t            getPrimitiveCount() const;

    void                createDescriptorPool(VkDevice* device, uint32_t imageCount);
    void                createDescriptorSet(VkDevice* device, uint32_t imageCount);

    float animationTimer = 0.0f;
    uint32_t animationIndex = 0;
    uint32_t newAnimationIndex;
    bool changeAnimationFlag = false;
    float startTimer;
    float changeAnimationTime;
};

#endif // OBJECT_H
