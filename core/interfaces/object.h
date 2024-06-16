#ifndef OBJECT_H
#define OBJECT_H

#include <vulkan.h>
#include <vector>
#include <vector.h>

#include <vkdefault.h>

namespace moon::utils { struct PhysicalDevice;}

namespace moon::interfaces {

class Model;

enum ObjectType : uint8_t {
    base = 0x1,
    skybox = 0x2
};

enum ObjectProperty : uint8_t {
    non = 0x0,
    outlining = 1<<4
};

class Object
{
protected:
    bool enable{true};
    bool enableShadow{true};

    uint32_t firstPrimitive{0};
    uint32_t primitiveCount{0};

    struct Outlining{
        bool Enable{false};
        float Width{0.0f};
        moon::math::Vector<float,4> Color{0.0f};
    }outlining;

    Model* pModel{nullptr};
    uint32_t firstInstance{0};
    uint32_t instanceCount{1};

    uint8_t pipelineBitMask{0};

    moon::utils::vkDefault::DescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool                descriptorPool{VK_NULL_HANDLE};
    std::vector<VkDescriptorSet>    descriptors;

public:
    virtual ~Object(){};

    void setModel(Model* model, uint32_t firstInstance = 0, uint32_t instanceCount = 1);
    Model* getModel();
    uint32_t getInstanceNumber(uint32_t imageNumber) const;

    void setEnable(const bool& enable);
    void setEnableShadow(const bool& enable);
    bool getEnable() const;
    bool getEnableShadow() const;

    void setOutlining(
        const bool& enable,
        const float& width = 0,
        const moon::math::Vector<float,4>& color = {0.0f});
    bool getOutliningEnable() const;
    float getOutliningWidth() const;
    moon::math::Vector<float,4> getOutliningColor() const;

    bool comparePrimitive(uint32_t primitive);
    void setFirstPrimitive(uint32_t firstPrimitive);
    void setPrimitiveCount(uint32_t primitiveCount);
    uint32_t getFirstPrimitive() const;
    uint32_t getPrimitiveCount() const;

    uint8_t getPipelineBitMask() const;
    const std::vector<VkDescriptorSet>& getDescriptorSet() const;

    virtual void destroy(
        VkDevice device) = 0;

    virtual void create(
        const moon::utils::PhysicalDevice& device,
        VkCommandPool commandPool,
        uint32_t imageCount) = 0;

    virtual void update(
        uint32_t frameNumber,
        VkCommandBuffer commandBuffer) = 0;

    static moon::utils::vkDefault::DescriptorSetLayout createDescriptorSetLayout(VkDevice device);
    static moon::utils::vkDefault::DescriptorSetLayout createSkyboxDescriptorSetLayout(VkDevice device);
};

}
#endif // OBJECT_H
