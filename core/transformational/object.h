#ifndef OBJECT_H
#define OBJECT_H

#include <libs/vulkan/vulkan.h>
#include "transformational.h"
#include "libs/quaternion.h"
#include "core/texture.h"

#include <string>

struct UniformBuffer
{
    alignas(16) glm::mat4x4 modelMatrix;
    alignas(16) glm::vec4   constantColor;
    alignas(16) glm::vec4   colorFactor;
    alignas(16) glm::vec4   bloomColor;
    alignas(16) glm::vec4   bloomFactor;
};

struct gltfModel;
struct Node;
struct Material;

class object : public transformational
{
private:
    bool                            enable{true};
    bool                            enableShadow{true};

    gltfModel**                     pModel;
    uint32_t                        modelCount{0};

    quaternion<float>               translation{0.0f,0.0f,0.0f,0.0f};
    quaternion<float>               rotation{1.0f,0.0f,0.0f,0.0f};
    glm::vec3                       scaling{1.0f,1.0f,1.0f};
    glm::mat4x4                     globalTransformation{1.0f};
    glm::mat4x4                     modelMatrix{1.0f};

    glm::vec4                       colorFactor{1.0f};
    glm::vec4                       constantColor{0.0f};
    glm::vec4                       bloomFactor{1.0f};
    glm::vec4                       bloomColor{0.0f};

    uint32_t                        firstPrimitive;
    uint32_t                        primitiveCount{0};

    struct Outlining{
        bool                        Enable{false};
        float                       Width{0.0f};
        glm::vec4                   Color{0.0f};
    }outlining;

    void updateModelMatrix();

protected:
    VkDescriptorSetLayout           descriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorPool                descriptorPool{VK_NULL_HANDLE};
    std::vector<VkDescriptorSet>    descriptors;

    struct buffer
    {
        VkBuffer instance{VK_NULL_HANDLE};
        VkDeviceMemory memory{VK_NULL_HANDLE};
        bool updateFlag{true};
    };
    std::vector<buffer> uniformBuffers;
    std::vector<buffer> uniformBuffersDevice;

public:
    object();
    object(uint32_t modelCount, gltfModel** model);
    ~object();
    void destroyUniformBuffers(VkDevice* device);
    void destroy(VkDevice* device);

    uint8_t                         getPipelineBitMask() const;

    VkDescriptorPool&               getDescriptorPool();
    std::vector<VkDescriptorSet>&   getDescriptorSet();

    void createUniformBuffers(VkPhysicalDevice* physicalDevice, VkDevice* device, uint32_t imageCount);
    void updateUniformBuffer(VkDevice device, VkCommandBuffer commandBuffer, uint32_t frameNumber);
    void updateAnimation(uint32_t imageNumber);

    void                setGlobalTransform(const glm::mat4& transform);
    void                translate(const glm::vec3& translate);
    void                rotate(const float& ang, const glm::vec3& ax);
    void                scale(const glm::vec3& scale);
    void                setPosition(const glm::vec3& translate);

    void                setModel(gltfModel** model3D);
    void                setConstantColor(const glm::vec4 & color);
    void                setColorFactor(const glm::vec4 & color);
    void                setBloomColor(const glm::vec4 & color);
    void                setBloomFactor(const glm::vec4 & color);

    glm::mat4x4         getModelMatrix() const;
    gltfModel*          getModel(uint32_t index);
    glm::vec4           getConstantColor() const;
    glm::vec4           getColorFactor() const;

    void                setEnable(const bool& enable);
    void                setEnableShadow(const bool& enable);
    bool                getEnable() const;
    bool                getEnableShadow() const;

    void                setOutliningEnable(const bool& enable);
    void                setOutliningWidth(const float& width);
    void                setOutliningColor(const glm::vec4& color);

    bool                getOutliningEnable() const;
    float               getOutliningWidth() const;
    glm::vec4           getOutliningColor() const;

    void                setFirstPrimitive(uint32_t firstPrimitive);
    void                setPrimitiveCount(uint32_t primitiveCount);
    void                resetPrimitiveCount();
    void                increasePrimitiveCount();

    bool                comparePrimitive(uint32_t primitive);
    uint32_t            getFirstPrimitive() const;
    uint32_t            getPrimitiveCount() const;

    void                createDescriptorPool(VkDevice* device, uint32_t imageCount);
    void                createDescriptorSet(VkDevice* device, uint32_t imageCount);

    float animationTimer{0.0f};
    uint32_t animationIndex{0};
    uint32_t newAnimationIndex;
    bool changeAnimationFlag{false};
    float startTimer;
    float changeAnimationTime;
};

class skyboxObject : public object{
private:
    cubeTexture* texture{nullptr};
public:
    skyboxObject(const std::vector<std::string>& TEXTURE_PATH);
    ~skyboxObject();

    void translate(const glm::vec3& translate);

    void createTexture(VkPhysicalDevice* physicalDevice, VkDevice* device, VkQueue* queue, VkCommandPool* commandPool);
    void destroyTexture(VkDevice* device);

    void createDescriptorPool(VkDevice* device, uint32_t imageCount);
    void createDescriptorSet(VkDevice* device, uint32_t imageCount);
};

#endif // OBJECT_H
