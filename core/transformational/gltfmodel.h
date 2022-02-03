#ifndef GLTFMODEL_H
#define GLTFMODEL_H

#include <iostream>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <vector>

#include "core/vulkanCore.h"
#include "core/texture.h"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <libs/glm/glm/glm.hpp>
#include <libs/glm/glm/gtc/matrix_transform.hpp>
#include <libs/glm/glm/gtc/type_ptr.hpp>
#include <libs/glm/glm/gtx/string_cast.hpp>

#include "libs/tinygltf-master/tiny_gltf.h"

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
    VkApplication* app;
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
    Mesh(VkApplication* app, glm::mat4 matrix);
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
    VkApplication* app;

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

        static VkVertexInputBindingDescription getBindingDescription()
        {
            VkVertexInputBindingDescription bindingDescription{};
            bindingDescription.binding = 0;                                 //является индексом привязки, оптсываемым данной структурой
                                                                            //каждый конверей может обратиться к определённому числу привлязок вершинных буферов,
                                                                            //и их индексы не обязательно должы быть непрерывными. Также необязательно описывать
                                                                            //каждую привязку в заданном кнвейере до тех пор, пока каждая используемая привязка описана
            bindingDescription.stride = sizeof(Vertex);                     //Каждая привязка может рассматриваться как массив структур, размещенных в буфере. Шаг массива stride - рассторяние между началами струкур,
                                                                            //измеряемое в байтах. Если вершинные данные задаются как массив структур, то, по сути, параметр stride содержит размер стуктуры, жаде если шейдер не будет использовать каждого члена этой стуктуры
            bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;     //Vulkan может обращаться в этот массив либо по индексу вершин, либо по индексу экземпляра в режиме дублирования геометрии.
                                                                            //Это задаётся в данном поле либо как VK_VERTEX_INPUT_RATE_VERTEX, либо как VK_VERTEX_INPUT_RATE_INSTANCE

            return bindingDescription;
        }

        static std::array<VkVertexInputAttributeDescription, 8> getAttributeDescriptions()
        {
            std::array<VkVertexInputAttributeDescription, 8> attributeDescriptions{};

            attributeDescriptions[0].binding = 0;                           //привязка к которой буфер привязан и из которого этот атрибут берёт данные
            attributeDescriptions[0].location = 0;                          //положение которое используется для обращения к атрибуту из вершинного шейдера
            attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;   //формат вершинных данных
            attributeDescriptions[0].offset = offsetof(Vertex, pos);        //смещение внутри каждой структуры

            attributeDescriptions[1].binding = 0;
            attributeDescriptions[1].location = 1;
            attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
            attributeDescriptions[1].offset = offsetof(Vertex, normal);

            attributeDescriptions[2].binding = 0;
            attributeDescriptions[2].location = 2;
            attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
            attributeDescriptions[2].offset = offsetof(Vertex, uv0);

            attributeDescriptions[3].binding = 0;
            attributeDescriptions[3].location = 3;
            attributeDescriptions[3].format = VK_FORMAT_R32G32_SFLOAT;
            attributeDescriptions[3].offset = offsetof(Vertex, uv1);

            attributeDescriptions[4].binding = 0;
            attributeDescriptions[4].location = 4;
            attributeDescriptions[4].format = VK_FORMAT_R32G32B32A32_SFLOAT;
            attributeDescriptions[4].offset = offsetof(Vertex, joint0);

            attributeDescriptions[5].binding = 0;
            attributeDescriptions[5].location = 5;
            attributeDescriptions[5].format = VK_FORMAT_R32G32B32A32_SFLOAT;
            attributeDescriptions[5].offset = offsetof(Vertex, weight0);

            attributeDescriptions[6].binding = 0;
            attributeDescriptions[6].location = 6;
            attributeDescriptions[6].format = VK_FORMAT_R32G32B32_SFLOAT;
            attributeDescriptions[6].offset = offsetof(Vertex, tangent);

            attributeDescriptions[7].binding = 0;
            attributeDescriptions[7].location = 7;
            attributeDescriptions[7].format = VK_FORMAT_R32G32B32_SFLOAT;
            attributeDescriptions[7].offset = offsetof(Vertex, bitangent);

            return attributeDescriptions;
        }

        bool operator==(const Vertex& other) const
        {
            return pos == other.pos;
        }
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

    glm::mat4 aabb;

    std::vector<Node*> nodes;
    std::vector<Node*> linearNodes;

    std::vector<Skin*> skins;

    std::vector<texture> textures;
    std::vector<textureSampler> textureSamplers;
    std::vector<Material> materials;
    std::vector<Animation> animations;
    std::vector<std::string> extensions;

    struct Dimensions {
        glm::vec3 min = glm::vec3(FLT_MAX);
        glm::vec3 max = glm::vec3(-FLT_MAX);
    } dimensions;

    void destroy(VkDevice device);
    void loadNode(Node* parent, const tinygltf::Node& node, uint32_t nodeIndex, const tinygltf::Model& model, std::vector<uint32_t>& indexBuffer, std::vector<Vertex>& vertexBuffer, float globalscale);
    void loadSkins(tinygltf::Model& gltfModel);
    void loadTextures(tinygltf::Model& gltfModel);
    VkSamplerAddressMode getVkWrapMode(int32_t wrapMode);
    VkFilter getVkFilterMode(int32_t filterMode);
    void loadTextureSamplers(tinygltf::Model& gltfModel);
    void loadMaterials(tinygltf::Model& gltfModel);
    void loadAnimations(tinygltf::Model& gltfModel);
    void loadFromFile(std::string filename, VkApplication* app, float scale = 1.0f);
    void drawNode(Node* node, VkCommandBuffer commandBuffer);
    void draw(VkCommandBuffer commandBuffer);
    void calculateBoundingBox(Node* node, Node* parent);
    void getSceneDimensions();
    void updateAnimation(uint32_t index, float time);
    void changeAnimation(uint32_t oldIndex, uint32_t newIndex, float startTime, float time, float changeAnimationTime);
    Node* findNode(Node* parent, uint32_t index);
    Node* nodeFromIndex(uint32_t index);
    void calculateTangent(std::vector<Vertex>& vertexBuffer, std::vector<uint32_t>& indexBuffer);
};

struct PushConstBlockMaterial
{
    glm::vec4 baseColorFactor;
    glm::vec4 emissiveFactor;
    glm::vec4 diffuseFactor;
    glm::vec4 specularFactor;
    float workflow;
    int colorTextureSet;
    int PhysicalDescriptorTextureSet;
    int normalTextureSet;
    int occlusionTextureSet;
    int emissiveTextureSet;
    float metallicFactor;
    float roughnessFactor;
    float alphaMask;
    float alphaMaskCutoff;
};

enum PBRWorkflows{ PBR_WORKFLOW_METALLIC_ROUGHNESS = 0, PBR_WORKFLOW_SPECULAR_GLOSINESS = 1 };

#endif // GLTFMODEL_H
