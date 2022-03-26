#ifndef OBJECT_H
#define OBJECT_H

#include "core/texture.h"
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
    gltfModel*                      model;
    texture*                        emptyTexture;

    VkDescriptorSetLayout*          uniformBufferSetLayout;
    VkDescriptorSetLayout*          uniformBlockSetLayout;
    VkDescriptorSetLayout*          materialSetLayout;
    VkDescriptorPool                descriptorPool;
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

    void updateModelMatrix();
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

    void setModel(gltfModel* model3D);
    void setEmptyTexture(texture* emptyTexture);
    void setVisibilityDistance(float visibilityDistance);
    void setColor(const glm::vec4 & color);

    void createUniformBuffers(uint32_t imageCount);
    void updateUniformBuffer(uint32_t currentImage);

    void setDescriptorSetLayouts(descriptorSetLayouts setLayouts);
    void createDescriptorPool(uint32_t imageCount);
    void createDescriptorSet(uint32_t imageCount);
    void createNodeDescriptorSet(Node* node);
    void createMaterialDescriptorSet(Material* material);


    void updateAnimation();

    gltfModel*                      getModel();

    glm::mat4x4&                    ModelMatrix();
    glm::mat4x4&                    Transformation();
    glm::vec3&                      Translate();
    glm::quat&                      Rotate();
    glm::vec3&                      Scale();

    float                           getVisibilityDistance();
    glm::vec4                       getColor();

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
