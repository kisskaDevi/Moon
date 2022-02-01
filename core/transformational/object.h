#ifndef OBJECT_H
#define OBJECT_H

#include "core/texture.h"
#include "transformational.h"
#include <array>
#include <vector>

struct UniformBuffer
{
    alignas(16) glm::mat4x4 modelMatrix;
};

struct PushConstant
{
    alignas(16) glm::mat4x4 modelMatrix;
};


struct gltfModel;
struct Node;
struct Material;

struct objectInfo
{
    gltfModel*                      model;
    texture*                        emptyTexture;
};

struct descriptorSetLayouts
{
    VkDescriptorSetLayout*          uniformBufferSetLayout;
    VkDescriptorSetLayout*          uniformBlockSetLayout;
    VkDescriptorSetLayout*          materialSetLayout;
};

class object : public transformational
{
private:
    VkApplication*                  app;
    gltfModel*                      m_model;
    texture*                        m_emptyTexture;

    VkDescriptorSetLayout*          m_uniformBufferSetLayout;
    VkDescriptorSetLayout*          m_uniformBlockSetLayout;
    VkDescriptorSetLayout*          m_materialSetLayout;
    VkDescriptorPool                m_descriptorPool;
    std::vector<VkDescriptorSet>    m_descriptors;

    glm::vec3                       m_translate;
    glm::quat                       m_rotate;
    glm::vec3                       m_scale;
    glm::mat4x4                     m_globalTransform;
    glm::mat4x4                     modelMatrix;
    float                           visibilityDistance = 10.0f;

    std::vector<VkBuffer>           m_uniformBuffers;
    std::vector<VkDeviceMemory>     m_uniformBuffersMemory;

public:
    object(VkApplication* app);
    object(VkApplication* app, objectInfo info);
    object(VkApplication* app, gltfModel* model3D);
    ~object();
    void destroyUniformBuffers();
    void destroyDescriptorPools();

    void setGlobalTransform(const glm::mat4& transform);
    void translate(const glm::vec3& translate);
    void rotate(const float& ang, const glm::vec3& ax);
    void scale(const glm::vec3& scale);

    void updateModelMatrix();
    void updateAnimation();

    void setModel(gltfModel* model3D);
    void setEmptyTexture(texture* emptyTexture);
    void setVisibilityDistance(float visibilityDistance);

    void createUniformBuffers(uint32_t imageCount);
    void updateUniformBuffer(uint32_t currentImage);

    void setDescriptorSetLayouts(descriptorSetLayouts setLayouts);
    void createDescriptorPool(uint32_t imageCount);
    void createDescriptorSet(uint32_t imageCount);
    void createNodeDescriptorSet(Node* node);
    void createMaterialDescriptorSet(Material* material);

    gltfModel*                      getModel();

    glm::mat4x4                     getTransformation();
    float                           getVisibilityDistance();

    VkDescriptorPool&               getDescriptorPool();
    std::vector<VkDescriptorSet>&   getDescriptorSet();

    float animationTimer = 0.0f;
    uint32_t animationIndex = 0;
    uint32_t newAnimationIndex;
    bool changeAnimationFlag = false;
    float startTimer;
    float changeAnimationTime;
};

#endif // OBJECT_H
