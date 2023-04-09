#ifndef MODEL_H
#define MODEL_H

#include <vulkan.h>
#include <glm.hpp>

#include <vector>

class texture;

struct Material {
    enum AlphaMode{ ALPHAMODE_OPAQUE, ALPHAMODE_MASK, ALPHAMODE_BLEND };
    AlphaMode alphaMode = ALPHAMODE_OPAQUE;
    float alphaCutoff{1.0f};
    float metallicFactor{1.0f};
    float roughnessFactor{1.0f};
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    glm::vec4 emissiveFactor = glm::vec4(1.0f);
    texture*   baseColorTexture;
    texture*   metallicRoughnessTexture;
    texture*   normalTexture;
    texture*   occlusionTexture;
    texture*   emissiveTexture;
    struct TexCoordSets {
        uint8_t baseColor{0};
        uint8_t metallicRoughness{0};
        uint8_t specularGlossiness{0};
        uint8_t normal{0};
        uint8_t occlusion{0};
        uint8_t emissive{0};
    } texCoordSets;
    struct Extension {
        texture* specularGlossinessTexture;
        texture* diffuseTexture;
        glm::vec4 diffuseFactor = glm::vec4(1.0f);
        glm::vec3 specularFactor = glm::vec3(0.0f);
    } extension;
    struct PbrWorkflows {
        bool metallicRoughness = true;
        bool specularGlossiness = false;
    } pbrWorkflows;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
};

struct MaterialBlock
{
    alignas(16) glm::vec4   baseColorFactor;
    alignas(16) glm::vec4   emissiveFactor;
    alignas(16) glm::vec4   diffuseFactor;
    alignas(16) glm::vec4   specularFactor;
    alignas(4)  float       workflow;
    alignas(4)  int         colorTextureSet;
    alignas(4)  int         PhysicalDescriptorTextureSet;
    alignas(4)  int         normalTextureSet;
    alignas(4)  int         occlusionTextureSet;
    alignas(4)  int         emissiveTextureSet;
    alignas(4)  float       metallicFactor;
    alignas(4)  float       roughnessFactor;
    alignas(4)  float       alphaMask;
    alignas(4)  float       alphaMaskCutoff;
    alignas(4)  uint32_t    primitive;
};

enum PBRWorkflows{ PBR_WORKFLOW_METALLIC_ROUGHNESS = 0, PBR_WORKFLOW_SPECULAR_GLOSINESS = 1 };

class model
{
public:
    struct Vertex{
        glm::vec3 pos;
        glm::vec3 normal;
        glm::vec2 uv0;
        glm::vec2 uv1;
        glm::vec4 joint0;
        glm::vec4 weight0;
        glm::vec3 tangent;
        glm::vec3 bitangent;

        static VkVertexInputBindingDescription getBindingDescription();
        static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions();
        bool operator==(const Vertex& other) const {return pos == other.pos;}
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
