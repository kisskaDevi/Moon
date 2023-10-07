#ifndef BASEOBJECT_H
#define BASEOBJECT_H

#include <vulkan.h>
#include <filesystem>

#include "transformational.h"
#include "quaternion.h"
#include "texture.h"
#include "buffer.h"
#include "object.h"
#include "matrix.h"

struct UniformBuffer
{
    alignas(16) matrix<float,4,4> modelMatrix;
    alignas(16) vector<float,4>   constantColor;
    alignas(16) vector<float,4>   colorFactor;
    alignas(16) vector<float,4>   bloomColor;
    alignas(16) vector<float,4>   bloomFactor;
};

class model;

class baseObject : public object, public transformational
{
private:
    bool                            enable{true};
    bool                            enableShadow{true};

    model*                          pModel;

    quaternion<float>               translation{0.0f,0.0f,0.0f,0.0f};
    quaternion<float>               rotation{1.0f,0.0f,0.0f,0.0f};
    vector<float,3>                 scaling{1.0f,1.0f,1.0f};
    matrix<float,4,4>               globalTransformation{1.0f};
    matrix<float,4,4>               modelMatrix{1.0f};

    vector<float,4>                 colorFactor{1.0f};
    vector<float,4>                 constantColor{0.0f};
    vector<float,4>                 bloomFactor{1.0f};
    vector<float,4>                 bloomColor{0.0f};

    uint32_t                        firstPrimitive{0};
    uint32_t                        primitiveCount{0};

    struct Outlining{
        bool                        Enable{false};
        float                       Width{0.0f};
        vector<float,4>             Color{0.0f};
    }outlining;

    uint32_t                        firstInstance{0};
    uint32_t                        instanceCount{1};

protected:
    VkDescriptorSetLayout           descriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorPool                descriptorPool{VK_NULL_HANDLE};
    std::vector<VkDescriptorSet>    descriptors;

    std::vector<buffer> uniformBuffersHost;
    std::vector<buffer> uniformBuffersDevice;

private:
    void updateUniformBuffersFlags(std::vector<buffer>& uniformBuffers);
    void updateModelMatrix();
public:
    baseObject() = default;
    baseObject(model* model, uint32_t firstInstance = 0, uint32_t instanceCount = 1);
    ~baseObject() = default;

    void destroy(VkDevice device) override;

    uint8_t                         getPipelineBitMask() const override;

    VkDescriptorPool&               getDescriptorPool();
    std::vector<VkDescriptorSet>&   getDescriptorSet() override;

    void createUniformBuffers(VkPhysicalDevice physicalDevice, VkDevice device, uint32_t imageCount) override;
    void updateUniformBuffer(VkCommandBuffer commandBuffer, uint32_t frameNumber) override;
    void updateAnimation(uint32_t imageNumber);

    baseObject&         setGlobalTransform(const matrix<float,4,4>& transform) override;
    baseObject&         translate(const vector<float,3>& translate) override;
    baseObject&         rotate(const float& ang, const vector<float,3>& ax) override;
    baseObject&         rotate(const quaternion<float>& quat);
    baseObject&         scale(const vector<float,3>& scale) override;
    baseObject&         setTranslation(const vector<float,3>& translate);

    void                setModel(model* model, uint32_t firstInstance = 0, uint32_t instanceCount = 1);
    void                setConstantColor(const vector<float,4> & color);
    void                setColorFactor(const vector<float,4> & color);
    void                setBloomColor(const vector<float,4> & color);
    void                setBloomFactor(const vector<float,4> & color);

    matrix<float,4,4>   getModelMatrix() const;
    model*              getModel() override;
    uint32_t            getInstanceNumber(uint32_t imageNumber) const override;
    vector<float,4>     getConstantColor() const;
    vector<float,4>     getColorFactor() const;

    void                setEnable(const bool& enable);
    void                setEnableShadow(const bool& enable);
    bool                getEnable() const override;
    bool                getEnableShadow() const override;

    void                setOutlining(const bool& enable, const float& width = 0, const vector<float,4>& color = {0.0f});
    bool                getOutliningEnable() const override;
    float               getOutliningWidth() const override;
    vector<float,4>     getOutliningColor() const override;

    void                setFirstPrimitive(uint32_t firstPrimitive) override;
    void                setPrimitiveCount(uint32_t primitiveCount) override;
    void                resetPrimitiveCount() override;
    void                increasePrimitiveCount();

    bool                comparePrimitive(uint32_t primitive);
    uint32_t            getFirstPrimitive() const override;
    uint32_t            getPrimitiveCount() const;

    void                createDescriptorPool(VkDevice device, uint32_t imageCount) override;
    void                createDescriptorSet(VkDevice device, uint32_t imageCount) override;

    cubeTexture* getTexture() override {return nullptr;};

    float animationTimer{0.0f};
    uint32_t animationIndex{0};
    uint32_t newAnimationIndex;
    bool changeAnimationFlag{false};
    float startTimer;
    float changeAnimationTime;
};

class skyboxObject : public baseObject{
private:
    cubeTexture* texture{nullptr};
public:
    skyboxObject(const std::vector<std::filesystem::path>& TEXTURE_PATH);
    ~skyboxObject();

    skyboxObject& translate(const vector<float,3>& translate) override;

    uint8_t getPipelineBitMask() const override;
    cubeTexture* getTexture() override;

    void createDescriptorPool(VkDevice device, uint32_t imageCount) override;
    void createDescriptorSet(VkDevice device, uint32_t imageCount) override;
};

#endif // BASEOBJECT_H
