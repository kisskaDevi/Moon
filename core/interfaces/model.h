#ifndef MODEL_H
#define MODEL_H

#include <vulkan.h>
#include <vector>
#include <vector.h>

class texture;

struct Material {
    enum AlphaMode{ ALPHAMODE_OPAQUE, ALPHAMODE_MASK, ALPHAMODE_BLEND };
    AlphaMode alphaMode = ALPHAMODE_OPAQUE;
    float alphaCutoff{1.0f};
    float metallicFactor{1.0f};
    float roughnessFactor{1.0f};
    vector<float,4> baseColorFactor{1.0f};
    vector<float,4> emissiveFactor{1.0f};
    texture*   baseColorTexture{nullptr};
    texture*   metallicRoughnessTexture{nullptr};
    texture*   normalTexture{nullptr};
    texture*   occlusionTexture{nullptr};
    texture*   emissiveTexture{nullptr};
    struct TexCoordSets {
        uint8_t baseColor{0};
        uint8_t metallicRoughness{0};
        uint8_t specularGlossiness{0};
        uint8_t normal{0};
        uint8_t occlusion{0};
        uint8_t emissive{0};
    } texCoordSets;
    struct Extension {
        texture* specularGlossinessTexture{nullptr};
        texture* diffuseTexture{nullptr};
        vector<float,4> diffuseFactor{1.0f};
        vector<float,3> specularFactor{0.0f};
    } extension;
    struct PbrWorkflows {
        bool metallicRoughness = true;
        bool specularGlossiness = false;
    } pbrWorkflows;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
};

struct MaterialBlock
{
    alignas(16) vector<float,4>   baseColorFactor{0.0f};
    alignas(16) vector<float,4>   emissiveFactor{0.0f};
    alignas(16) vector<float,4>   diffuseFactor{0.0f};
    alignas(16) vector<float,4>   specularFactor{0.0f};
    alignas(4)  float       workflow{0.0f};
    alignas(4)  int         colorTextureSet{-1};
    alignas(4)  int         PhysicalDescriptorTextureSet{-1};
    alignas(4)  int         normalTextureSet{-1};
    alignas(4)  int         occlusionTextureSet{-1};
    alignas(4)  int         emissiveTextureSet{-1};
    alignas(4)  float       metallicFactor{0.0f};
    alignas(4)  float       roughnessFactor{0.0f};
    alignas(4)  float       alphaMask{0.0f};
    alignas(4)  float       alphaMaskCutoff{0.0f};
    alignas(4)  uint32_t    primitive;
};

enum PBRWorkflows{ PBR_WORKFLOW_METALLIC_ROUGHNESS = 0, PBR_WORKFLOW_SPECULAR_GLOSINESS = 1 };

class model
{
public:
    struct Vertex{
        alignas(16) vector<float,3> pos{0.0f};
        alignas(16) vector<float,3> normal{0.0f};
        alignas(16) vector<float,2> uv0{0.0f};
        alignas(16) vector<float,2> uv1{0.0f};
        alignas(16) vector<float,4> joint0{0.0f};
        alignas(16) vector<float,4> weight0{0.0f};
        alignas(16) vector<float,3> tangent{0.0f};
        alignas(16) vector<float,3> bitangent{0.0f};

        static VkVertexInputBindingDescription getBindingDescription();
        static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions();
    };


    virtual void destroy(VkDevice device) = 0;
    virtual void destroyStagingBuffer(VkDevice device) = 0;

    virtual const VkBuffer* getVertices() const = 0;
    virtual const VkBuffer* getIndices() const = 0;

    virtual void loadFromFile(VkPhysicalDevice physicalDevice, VkDevice device, VkCommandBuffer commandBuffer) = 0;

    virtual bool hasAnimation(uint32_t frameIndex) const = 0;
    virtual float animationStart(uint32_t frameIndex, uint32_t index) const = 0;
    virtual float animationEnd(uint32_t frameIndex, uint32_t index) const = 0;
    virtual void updateAnimation(uint32_t frameIndex, uint32_t index, float time) = 0;
    virtual void changeAnimation(uint32_t frameIndex, uint32_t oldIndex, uint32_t newIndex, float startTime, float time, float changeAnimationTime) = 0;

    virtual void createDescriptorPool(VkDevice device) = 0;
    virtual void createDescriptorSet(VkDevice device, texture* emptyTexture) = 0;

    virtual void render(uint32_t frameIndex, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t& primitiveCount, uint32_t pushConstantSize, uint32_t pushConstantOffset, void* pushConstant) = 0;

    static void createNodeDescriptorSetLayout(
            VkDevice                        device,
            VkDescriptorSetLayout*          descriptorSetLayout);

    static void createMaterialDescriptorSetLayout(
            VkDevice                        device,
            VkDescriptorSetLayout*          descriptorSetLayout);
};

#endif // MODEL_H
