#ifndef PLYMODEL_H
#define PLYMODEL_H

#include "model.h"
#include "texture.h"
#include "buffer.h"
#include "matrix.h"

#include <filesystem>

#define MAX_NUM_JOINTS 128u

class plyModel : public model{
private:
    std::filesystem::path filename;

    buffer vertices, indices;
    buffer vertexStaging, indexStaging;

    uint32_t indexCount{0};
    vector<float,3> maxSize{0.0f};

    VkDescriptorSetLayout           nodeDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout           materialDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool                descriptorPool = VK_NULL_HANDLE;

    Material material;
    MaterialBlock materialBlock{};

    class UniformBuffer : public buffer {
    public:
        UniformBuffer() = default;
        VkDescriptorSet descriptorSet{VK_NULL_HANDLE};
    } uniformBuffer;

    struct UniformBlock {
        matrix<float,4,4> mat;
        matrix<float,4,4> jointMatrix[MAX_NUM_JOINTS]{};
        float jointcount{0};
    } uniformBlock;

public:
    plyModel(std::filesystem::path filename,
             vector<float, 4> baseColorFactor = vector<float, 4>(0.5f,0.5f,0.5f,1.0f),
             vector<float, 4> diffuseFactor = vector<float, 4>(0.5f),
             vector<float, 4> specularFactor = vector<float, 4>(0.5f),
             float metallicFactor = 0.0f,
             float roughnessFactor = 0.5f,
             float workflow = 1.0f);

    ~plyModel() override;
    void destroy(VkDevice device) override;
    void destroyStagingBuffer(VkDevice device) override;

    const VkBuffer* getVertices() const override;
    const VkBuffer* getIndices() const override;
    const vector<float,3> getMaxSize() const;

    void loadFromFile(VkPhysicalDevice physicalDevice, VkDevice device, VkCommandBuffer commandBuffer) override;

    bool hasAnimation(uint32_t) const override {return false;}
    float animationStart(uint32_t, uint32_t) const override {return 0.0f;}
    float animationEnd(uint32_t, uint32_t) const override {return 0.0f;}
    void updateAnimation(uint32_t, uint32_t, float) override {};
    void changeAnimation(uint32_t, uint32_t, uint32_t, float, float, float) override {};

    void createDescriptorPool(VkDevice device) override;
    void createDescriptorSet(VkDevice device, texture* emptyTexture) override;

    void render(uint32_t frameIndex, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t& primitiveCount, uint32_t pushConstantSize, uint32_t pushConstantOffset, void* pushConstant) override;
};

#endif // PLYMODEL_H
