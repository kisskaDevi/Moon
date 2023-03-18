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

#include <string>
#include <vector>

#define MAX_NUM_JOINTS 130u

struct Node;

struct BoundingBox{
    glm::vec3 min;
    glm::vec3 max;
    bool valid = false;
    BoundingBox();
    BoundingBox(glm::vec3 min, glm::vec3 max);
    BoundingBox getAABB(glm::mat4 m);
};

struct Material
{
    enum AlphaMode{ ALPHAMODE_OPAQUE, ALPHAMODE_MASK, ALPHAMODE_BLEND };
    AlphaMode alphaMode = ALPHAMODE_OPAQUE;
    float alphaCutoff = 1.0f;
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    glm::vec4 emissiveFactor = glm::vec4(1.0f);
    texture*   baseColorTexture;
    texture*   metallicRoughnessTexture;
    texture*   normalTexture;
    texture*   occlusionTexture;
    texture*   emissiveTexture;
    struct TexCoordSets {
        uint8_t baseColor = 0;
        uint8_t metallicRoughness = 0;
        uint8_t specularGlossiness = 0;
        uint8_t normal = 0;
        uint8_t occlusion = 0;
        uint8_t emissive = 0;
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

struct Primitive
{
    uint32_t firstIndex;
    uint32_t indexCount;
    uint32_t vertexCount;
    Material &material;
    bool hasIndices;
    BoundingBox bb;
    Primitive(uint32_t firstIndex, uint32_t indexCount, uint32_t vertexCount, Material& material);
    void setBoundingBox(glm::vec3 min, glm::vec3 max);
};

struct Mesh
{
    std::vector<Primitive*> primitives;
    BoundingBox bb;
    BoundingBox aabb;
    struct UniformBuffer {
        VkBuffer buffer;
        VkDeviceMemory memory;
        VkDescriptorBufferInfo descriptor;
        VkDescriptorSet descriptorSet;
        void *mapped;
    } uniformBuffer;
    struct UniformBlock {
        glm::mat4 matrix;
        glm::mat4 jointMatrix[MAX_NUM_JOINTS]{};
        float jointcount{ 0 };
    } uniformBlock;
    Mesh(VkPhysicalDevice* physicalDevice, VkDevice* device, glm::mat4 matrix);
    void destroy(VkDevice* device);
    ~Mesh();
    void setBoundingBox(glm::vec3 min, glm::vec3 max);
};

struct Skin
{
    std::string name;
    Node *skeletonRoot = nullptr;
    std::vector<glm::mat4> inverseBindMatrices;
    std::vector<Node*> joints;
};

struct Node
{
    Node *parent;
    uint32_t index;
    std::vector<Node*> children;
    glm::mat4 matrix;
    std::string name;
    Mesh *mesh;
    Skin *skin;
    int32_t skinIndex = -1;
    glm::vec3 translation{};
    glm::vec3 scale{ 1.0f };
    glm::quat rotation{};
    BoundingBox bvh;
    BoundingBox aabb;
    glm::mat4 localMatrix();
    glm::mat4 getMatrix();
    void update();
    void destroy(VkDevice* device);
    ~Node();
};

struct AnimationChannel
{
    enum PathType { TRANSLATION, ROTATION, SCALE };
    PathType path;
    Node *node;
    uint32_t samplerIndex;
};

struct AnimationSampler
{
    enum InterpolationType { LINEAR, STEP, CUBICSPLINE };
    InterpolationType interpolation;
    std::vector<float> inputs;
    std::vector<glm::vec4> outputsVec4;
};

struct Animation
{
    std::string name;
    std::vector<AnimationSampler> samplers;
    std::vector<AnimationChannel> channels;
    float start = std::numeric_limits<float>::max();
    float end = std::numeric_limits<float>::min();
};

struct gltfModel
{
    std::string filename;

    struct Vertex
    {
        glm::vec3 pos;
        glm::vec3 normal;
        glm::vec2 uv0;
        glm::vec2 uv1;
        glm::vec4 joint0;
        glm::vec4 weight0;
        glm::vec3 tangent;
        glm::vec3 bitangent;

        static VkVertexInputBindingDescription getShadowBindingDescription();
        static std::array<VkVertexInputAttributeDescription, 3> getShadowAttributeDescriptions();
        static VkVertexInputBindingDescription getBindingDescription();
        static std::array<VkVertexInputAttributeDescription, 8> getAttributeDescriptions();
        bool operator==(const Vertex& other) const {return pos == other.pos;}
    };

    struct Vertices
    {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory;
    } vertices;

    struct Indices
    {
        int count;
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory;
    } indices;

    struct StagingBuffer {
        VkBuffer buffer;
        VkDeviceMemory memory;
    } vertexStaging, indexStaging;

    VkDescriptorSetLayout           nodeDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout           materialDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool                DescriptorPool = VK_NULL_HANDLE;

    glm::mat4                       aabb;

    std::vector<Node*>              nodes;
    std::vector<Node*>              linearNodes;

    std::vector<Skin*>              skins;
    std::vector<Animation>          animations;

    std::vector<texture>            textures;
    std::vector<textureSampler>     textureSamplers;
    std::vector<Material>           materials;
    std::vector<std::string>        extensions;

    struct Dimensions {
        glm::vec3 min = glm::vec3(FLT_MAX);
        glm::vec3 max = glm::vec3(-FLT_MAX);
    } dimensions;

    gltfModel(std::string filename);

    void destroy(VkDevice* device);
    void destroyStagingBuffer(VkDevice device);
    void loadNode(VkPhysicalDevice* physicalDevice, VkDevice* device, Node* parent, const tinygltf::Node& node, uint32_t nodeIndex, const tinygltf::Model& model, std::vector<uint32_t>& indexBuffer, std::vector<Vertex>& vertexBuffer, float globalscale);
    void loadSkins(tinygltf::Model& gltfModel);
    void loadTextures(VkPhysicalDevice physicalDevice, VkDevice device, VkCommandBuffer commandBuffer, tinygltf::Model& gltfModel);
    VkSamplerAddressMode getVkWrapMode(int32_t wrapMode);
    VkFilter getVkFilterMode(int32_t filterMode);
    void loadTextureSamplers(tinygltf::Model& gltfModel);
    void loadMaterials(tinygltf::Model& gltfModel);
    void loadAnimations(tinygltf::Model& gltfModel);
    void loadFromFile(VkPhysicalDevice physicalDevice, VkDevice device, VkCommandBuffer commandBuffer);
    void calculateBoundingBox(Node* node, Node* parent);
    void getSceneDimensions();
    void updateAnimation(uint32_t index, float time);
    void changeAnimation(uint32_t oldIndex, uint32_t newIndex, float startTime, float time, float changeAnimationTime);
    Node* findNode(Node* parent, uint32_t index);
    Node* nodeFromIndex(uint32_t index);
    void calculateTangent(std::vector<Vertex>& vertexBuffer, std::vector<uint32_t>& indexBuffer);
    void calculateNodeTangent(Node* node, std::vector<Vertex>& vertexBuffer, std::vector<uint32_t>& indexBuffer);

    void createDescriptorPool(VkDevice* device);
    void createDescriptorSet(VkDevice* device, texture* emptyTexture);
        void createNodeDescriptorSet(VkDevice* device, Node* node);
        void createMaterialDescriptorSet(VkDevice* device, Material* material, texture* emptyTexture);


    static void createNodeDescriptorSetLayout(
            VkDevice                        device,
            VkDescriptorSetLayout*          descriptorSetLayout);

    static void createMaterialDescriptorSetLayout(
            VkDevice                        device,
            VkDescriptorSetLayout*          descriptorSetLayout);
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

#endif // GLTFMODEL_H
