#ifndef OBJECT_H
#define OBJECT_H

#include <vulkan.h>
#include <vector>
#include <vector.h>

class model;
struct physicalDevice;

class object
{
protected:
    bool enable{true};
    bool enableShadow{true};

    uint32_t firstPrimitive{0};
    uint32_t primitiveCount{0};

    struct Outlining{
        bool Enable{false};
        float Width{0.0f};
        vector<float,4> Color{0.0f};
    }outlining;

    model* pModel{nullptr};
    uint32_t firstInstance{0};
    uint32_t instanceCount{1};

public:
    virtual ~object(){};

    void setModel(model* model, uint32_t firstInstance = 0, uint32_t instanceCount = 1);
    model* getModel();
    uint32_t getInstanceNumber(uint32_t imageNumber) const;

    void setEnable(const bool& enable);
    void setEnableShadow(const bool& enable);
    bool getEnable() const;
    bool getEnableShadow() const;

    void setOutlining(
        const bool& enable,
        const float& width = 0,
        const vector<float,4>& color = {0.0f});
    bool getOutliningEnable() const;
    float getOutliningWidth() const;
    vector<float,4> getOutliningColor() const;

    bool comparePrimitive(uint32_t primitive);
    void setFirstPrimitive(uint32_t firstPrimitive);
    void setPrimitiveCount(uint32_t primitiveCount);
    uint32_t getFirstPrimitive() const;
    uint32_t getPrimitiveCount() const;

    virtual void destroy(
        VkDevice device) = 0;

    virtual void create(
        physicalDevice device,
        VkCommandPool commandPool,
        uint32_t imageCount) = 0;

    virtual void update(
        uint32_t frameNumber,
        VkCommandBuffer commandBuffer) = 0;

    virtual uint8_t getPipelineBitMask() const = 0;
    virtual std::vector<VkDescriptorSet>& getDescriptorSet() = 0;

    static void createDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout);
    static void createSkyboxDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout);
};

#endif // OBJECT_H
