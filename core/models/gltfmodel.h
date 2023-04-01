#ifndef GLTFMODEL_H
#define GLTFMODEL_H

#include <vulkan.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
#include <gtx/string_cast.hpp>
#include "tiny_gltf.h"

#include "../utils/texture.h"
#include "../interfaces/model.h"

#include <string>
#include <vector>

#define MAX_NUM_JOINTS 128u

struct BoundingBox{
    glm::vec3 min;
    glm::vec3 max;
    bool valid{false};

    BoundingBox() = default;
    BoundingBox(glm::vec3 min, glm::vec3 max);
    BoundingBox getAABB(glm::mat4 m) const;
};

struct buffer{
    VkBuffer       instance{VK_NULL_HANDLE};
    VkDeviceMemory memory{VK_NULL_HANDLE};
    void*          map{nullptr};

    void destroy(VkDevice device){
        if (instance != VK_NULL_HANDLE){ vkDestroyBuffer(device, instance, nullptr); instance = VK_NULL_HANDLE;}
        if (memory != VK_NULL_HANDLE){ vkFreeMemory(device, memory, nullptr); memory = VK_NULL_HANDLE;}
        if (map){ map = nullptr;}
    }
};

struct Mesh{
    struct Primitive{
        uint32_t firstIndex;
        uint32_t indexCount;
        uint32_t vertexCount;
        Material* material;
        bool hasIndices;
        BoundingBox bb;
        Primitive(uint32_t firstIndex, uint32_t indexCount, uint32_t vertexCount, Material* material);
    };

    struct UniformBuffer : public buffer {
        VkDescriptorSet descriptorSet;
    } uniformBuffer;

    struct UniformBlock {
        glm::mat4 matrix;
        glm::mat4 jointMatrix[MAX_NUM_JOINTS]{};
        float jointcount{0};
    } uniformBlock;

    std::vector<Primitive*> primitives;

    Mesh(VkPhysicalDevice physicalDevice, VkDevice device, glm::mat4 matrix);
    void destroy(VkDevice device);
    ~Mesh() = default;
};

struct Node;

struct Skin {
    std::vector<glm::mat4> inverseBindMatrices;
    std::vector<Node*> joints;
};

struct Node {
    uint32_t index;
    Node* parent{nullptr};
    Mesh* mesh{nullptr};
    Skin* skin;

    std::vector<Node*> children;

    glm::mat4 matrix;
    glm::vec3 translation{};
    glm::vec3 scale{1.0f};
    glm::quat rotation{};

    void update();
    void destroy(VkDevice device);
    size_t meshCount() const;
    ~Node() = default;
};

struct Animation
{
    struct AnimationChannel{
        enum PathType { TRANSLATION, ROTATION, SCALE };
        PathType path;
        Node* node;
        uint32_t samplerIndex;
    };

    struct AnimationSampler{
        enum InterpolationType { LINEAR, STEP, CUBICSPLINE };
        InterpolationType interpolation;
        std::vector<float> inputs;
        std::vector<glm::vec4> outputsVec4;
    };

    std::vector<AnimationSampler> samplers;
    std::vector<AnimationChannel> channels;
    float start = std::numeric_limits<float>::max();
    float end = std::numeric_limits<float>::min();
};

class gltfModel : public model
{
private:
    std::string filename{};

    buffer vertices, indices;
    buffer vertexStaging, indexStaging;

    VkDescriptorSetLayout           nodeDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout           materialDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool                descriptorPool = VK_NULL_HANDLE;

    std::vector<Node*>              nodes;
    std::vector<Skin*>              skins;
    std::vector<texture>            textures;
    std::vector<Material>           materials;
    std::vector<Animation>          animations;

    void loadNode(VkPhysicalDevice physicalDevice, VkDevice device, Node* parent, const tinygltf::Node& node, uint32_t nodeIndex, const tinygltf::Model& model, std::vector<uint32_t>& indexBuffer, std::vector<Vertex>& vertexBuffer);
    void loadSkins(tinygltf::Model& gltfModel);
    void loadTextures(VkPhysicalDevice physicalDevice, VkDevice device, VkCommandBuffer commandBuffer, tinygltf::Model& gltfModel);
    void loadMaterials(tinygltf::Model& gltfModel);
    void loadAnimations(tinygltf::Model& gltfModel);

    Node* nodeFromIndex(uint32_t index, const std::vector<Node*>& nodes);
public:
    gltfModel(std::string filename);

    void destroy(VkDevice device) override;
    void destroyStagingBuffer(VkDevice device) override;

    const VkBuffer* getVertices() const override;
    const VkBuffer* getIndices() const override;

    void loadFromFile(VkPhysicalDevice physicalDevice, VkDevice device, VkCommandBuffer commandBuffer) override;

    bool hasAnimation() const override;
    float animationStart(uint32_t index) const override;
    float animationEnd(uint32_t index) const override;
    void updateAnimation(uint32_t index, float time) override;
    void changeAnimation(uint32_t oldIndex, uint32_t newIndex, float startTime, float time, float changeAnimationTime) override;

    void createDescriptorPool(VkDevice device) override;
    void createDescriptorSet(VkDevice device, texture* emptyTexture) override;

    void render(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t& primitiveCount, uint32_t pushConstantSize, uint32_t pushConstantOffset, void* pushConstant) override;
};

#endif // GLTFMODEL_H
