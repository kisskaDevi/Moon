#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define STBI_MSC_SECURE_CRT

#include "gltfmodel.h"
#include "core/operations.h"

#include <iostream>
#include <fstream>
#include <stdlib.h>

/* BoundingBox */

BoundingBox::BoundingBox(){};

BoundingBox::BoundingBox(glm::vec3 min, glm::vec3 max): min(min), max(max){};

BoundingBox BoundingBox::getAABB(glm::mat4 m)
{
    glm::vec3 min = glm::vec3(m[3]);
    glm::vec3 max = min;
    glm::vec3 v0, v1;

    glm::vec3 right = glm::vec3(m[0]);
    v0 = right * this->min.x;
    v1 = right * this->max.x;
    min += glm::min(v0, v1);
    max += glm::max(v0, v1);

    glm::vec3 up = glm::vec3(m[1]);
    v0 = up * this->min.y;
    v1 = up * this->max.y;
    min += glm::min(v0, v1);
    max += glm::max(v0, v1);

    glm::vec3 back = glm::vec3(m[2]);
    v0 = back * this->min.z;
    v1 = back * this->max.z;
    min += glm::min(v0, v1);
    max += glm::max(v0, v1);

    return BoundingBox(min, max);
}

Primitive::Primitive(uint32_t firstIndex, uint32_t indexCount, uint32_t vertexCount, Material &material)
    : firstIndex(firstIndex), indexCount(indexCount), vertexCount(vertexCount), material(material)
{
    hasIndices = indexCount > 0;
}

void Primitive::setBoundingBox(glm::vec3 min, glm::vec3 max)
{
    bb.min = min;
    bb.max = max;
    bb.valid = true;
}

Mesh::Mesh(VkPhysicalDevice* physicalDevice, VkDevice* device, glm::mat4 matrix)
{
    this->uniformBlock.matrix = matrix;
    createBuffer(   physicalDevice,
                    device,
                    sizeof(uniformBlock),
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    uniformBuffer.buffer,
                    uniformBuffer.memory);
    vkMapMemory(*device, uniformBuffer.memory, 0, sizeof(uniformBlock), 0, &uniformBuffer.mapped);
    uniformBuffer.descriptor = { uniformBuffer.buffer, 0, sizeof(uniformBlock) };
};

void Mesh::destroy(VkDevice* device){
    vkDestroyBuffer(*device, uniformBuffer.buffer, nullptr);
    vkFreeMemory(*device, uniformBuffer.memory, nullptr);
    for (Primitive* p : primitives)
        delete p;
}

Mesh::~Mesh() {
}

void Mesh::setBoundingBox(glm::vec3 min, glm::vec3 max) {
    bb.min = min;
    bb.max = max;
    bb.valid = true;
}

glm::mat4 Node::localMatrix()
{
    return glm::translate(glm::mat4(1.0f), translation) * glm::mat4(rotation) * glm::scale(glm::mat4(1.0f), scale) * matrix;
}

glm::mat4 Node::getMatrix() {
    glm::mat4 m = localMatrix();
    Node *p = parent;
    while (p) {
        m = p->localMatrix() * m;
        p = p->parent;
    }
    return m;
}

void Node::update() {
    if (mesh)
    {
        glm::mat4 m = getMatrix();
        if (skin) {
            mesh->uniformBlock.matrix = m;
            // Update join matrices
            glm::mat4 inverseTransform = glm::inverse(m);
            size_t numJoints = std::min((uint32_t)skin->joints.size(), MAX_NUM_JOINTS);
            for (size_t i = 0; i < numJoints; i++) {
                Node *jointNode = skin->joints[i];
                glm::mat4 jointMat = jointNode->getMatrix() * skin->inverseBindMatrices[i];
                jointMat = inverseTransform * jointMat;
                mesh->uniformBlock.jointMatrix[i] = jointMat;
            }
            mesh->uniformBlock.jointcount = (float)numJoints;
            memcpy(mesh->uniformBuffer.mapped, &mesh->uniformBlock, sizeof(mesh->uniformBlock));
        } else {
            memcpy(mesh->uniformBuffer.mapped, &m, sizeof(glm::mat4));
        }
    }

    for (auto& child : children)
    {
        child->update();
    }
}

void Node::destroy(VkDevice *device)
{
    if (mesh) {
        mesh->destroy(device);
        delete mesh;
    }
    for (auto& child : children)
    {
        child->destroy(device);
        delete child;
    }

}

Node::~Node(){
}

// Model

gltfModel::gltfModel(std::string filename)
{
    this->filename = filename;
}

void gltfModel::destroy(VkDevice* device)
{
    if(DescriptorPool != VK_NULL_HANDLE){
        vkDestroyDescriptorPool(*device, DescriptorPool, nullptr);
        DescriptorPool = VK_NULL_HANDLE;
    }

    if (vertices.buffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(*device, vertices.buffer, nullptr);
        vkFreeMemory(*device, vertices.memory, nullptr);
        vertices.buffer = VK_NULL_HANDLE;
    }
    if (indices.buffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(*device, indices.buffer, nullptr);
        vkFreeMemory(*device, indices.memory, nullptr);
        indices.buffer = VK_NULL_HANDLE;
    }
    for (auto texture : textures)
    {
        texture.destroy(device);
    }
    textures.resize(0);
    textureSamplers.resize(0);
    for (auto node : nodes)
    {
        node->destroy(device);
        delete node;
    }
    if(nodeDescriptorSetLayout != VK_NULL_HANDLE){
        vkDestroyDescriptorSetLayout(*device, nodeDescriptorSetLayout,nullptr);
    }
    nodeDescriptorSetLayout = VK_NULL_HANDLE;
    if(materialDescriptorSetLayout != VK_NULL_HANDLE){
        vkDestroyDescriptorSetLayout(*device, materialDescriptorSetLayout,nullptr);
    }
    materialDescriptorSetLayout = VK_NULL_HANDLE;
    materials.resize(0);
    animations.resize(0);
    nodes.resize(0);
    linearNodes.resize(0);
    extensions.resize(0);
    for (auto skin : skins)
    {
        delete skin;
    }
    skins.resize(0);
};

void gltfModel::loadNode(VkPhysicalDevice* physicalDevice, VkDevice* device, Node *parent, const tinygltf::Node &node, uint32_t nodeIndex, const tinygltf::Model &model, std::vector<uint32_t>& indexBuffer, std::vector<Vertex>& vertexBuffer, float globalscale)
{
    Node *newNode = new Node{};
    newNode->index = nodeIndex;
    newNode->parent = parent;
    newNode->name = node.name;
    newNode->skinIndex = node.skin;
    newNode->matrix = glm::mat4(1.0f);

    // Generate local node matrix
    glm::vec3 translation = glm::vec3(0.0f);
    if (node.translation.size() == 3) {
        translation = glm::make_vec3(node.translation.data());
        newNode->translation = translation;
    }
    //glm::mat4 rotation = glm::mat4(1.0f);     //ADD
    if (node.rotation.size() == 4) {
        glm::quat q = glm::make_quat(node.rotation.data());
        newNode->rotation = glm::mat4(q);
    }
    glm::vec3 scale = glm::vec3(1.0f);
    if (node.scale.size() == 3) {
        scale = glm::make_vec3(node.scale.data());
        newNode->scale = scale;
    }
    if (node.matrix.size() == 16) {
        newNode->matrix = glm::make_mat4x4(node.matrix.data());
    };

    // Node with children
    if (node.children.size() > 0) {
        for (size_t i = 0; i < node.children.size(); i++) {
            loadNode(physicalDevice, device, newNode, model.nodes[node.children[i]], node.children[i], model, indexBuffer, vertexBuffer, globalscale);
        }
    }

    // Node contains mesh data
    if (node.mesh > -1) {
        const tinygltf::Mesh mesh = model.meshes[node.mesh];
        Mesh *newMesh = new Mesh(physicalDevice,device,newNode->matrix);
        for (size_t j = 0; j < mesh.primitives.size(); j++) {
            const tinygltf::Primitive &primitive = mesh.primitives[j];
            uint32_t indexStart = static_cast<uint32_t>(indexBuffer.size());
            uint32_t vertexStart = static_cast<uint32_t>(vertexBuffer.size());
            uint32_t indexCount = 0;
            uint32_t vertexCount = 0;
            glm::vec3 posMin{};
            glm::vec3 posMax{};
            bool hasSkin = false;
            bool hasIndices = primitive.indices > -1;
            // Vertices
            {
                const float *bufferPos = nullptr;
                const float *bufferNormals = nullptr;
                const float *bufferTexCoordSet0 = nullptr;
                const float *bufferTexCoordSet1 = nullptr;
                const void *bufferJoints = nullptr;
                const float *bufferWeights = nullptr;

                int posByteStride=0;        //  int posByteStride;
                int normByteStride=0;       //  int normByteStride;
                int uv0ByteStride=0;        //  int uv0ByteStride;
                int uv1ByteStride=0;        //  int uv1ByteStride;
                int jointByteStride=0;      //  int jointByteStride;
                int weightByteStride=0;     //  int weightByteStride;

                int jointComponentType=0;   //  int jointComponentType;

                // Position attribute is required
                assert(primitive.attributes.find("POSITION") != primitive.attributes.end());

                const tinygltf::Accessor &posAccessor = model.accessors[primitive.attributes.find("POSITION")->second];
                const tinygltf::BufferView &posView = model.bufferViews[posAccessor.bufferView];
                bufferPos = reinterpret_cast<const float *>(&(model.buffers[posView.buffer].data[posAccessor.byteOffset + posView.byteOffset]));
                posMin = glm::vec3(posAccessor.minValues[0], posAccessor.minValues[1], posAccessor.minValues[2]);
                posMax = glm::vec3(posAccessor.maxValues[0], posAccessor.maxValues[1], posAccessor.maxValues[2]);
                vertexCount = static_cast<uint32_t>(posAccessor.count);
                posByteStride = posAccessor.ByteStride(posView) ? (posAccessor.ByteStride(posView) / sizeof(float)) : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC3);

                if (primitive.attributes.find("NORMAL") != primitive.attributes.end()) {
                    const tinygltf::Accessor &normAccessor = model.accessors[primitive.attributes.find("NORMAL")->second];
                    const tinygltf::BufferView &normView = model.bufferViews[normAccessor.bufferView];
                    bufferNormals = reinterpret_cast<const float *>(&(model.buffers[normView.buffer].data[normAccessor.byteOffset + normView.byteOffset]));
                    normByteStride = normAccessor.ByteStride(normView) ? (normAccessor.ByteStride(normView) / sizeof(float)) : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC3);
                }

                if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end()) {
                    const tinygltf::Accessor &uvAccessor = model.accessors[primitive.attributes.find("TEXCOORD_0")->second];
                    const tinygltf::BufferView &uvView = model.bufferViews[uvAccessor.bufferView];
                    bufferTexCoordSet0 = reinterpret_cast<const float *>(&(model.buffers[uvView.buffer].data[uvAccessor.byteOffset + uvView.byteOffset]));
                    uv0ByteStride = uvAccessor.ByteStride(uvView) ? (uvAccessor.ByteStride(uvView) / sizeof(float)) : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC2);
                }
                if (primitive.attributes.find("TEXCOORD_1") != primitive.attributes.end()) {
                    const tinygltf::Accessor &uvAccessor = model.accessors[primitive.attributes.find("TEXCOORD_1")->second];
                    const tinygltf::BufferView &uvView = model.bufferViews[uvAccessor.bufferView];
                    bufferTexCoordSet1 = reinterpret_cast<const float *>(&(model.buffers[uvView.buffer].data[uvAccessor.byteOffset + uvView.byteOffset]));
                    uv1ByteStride = uvAccessor.ByteStride(uvView) ? (uvAccessor.ByteStride(uvView) / sizeof(float)) : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC2);
                }

                // Skinning
                // Joints
                if (primitive.attributes.find("JOINTS_0") != primitive.attributes.end()) {
                    const tinygltf::Accessor &jointAccessor = model.accessors[primitive.attributes.find("JOINTS_0")->second];
                    const tinygltf::BufferView &jointView = model.bufferViews[jointAccessor.bufferView];
                    bufferJoints = &(model.buffers[jointView.buffer].data[jointAccessor.byteOffset + jointView.byteOffset]);
                    jointComponentType = jointAccessor.componentType;
                    jointByteStride = jointAccessor.ByteStride(jointView) ? (jointAccessor.ByteStride(jointView) / tinygltf::GetComponentSizeInBytes(jointComponentType)) : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC4);
                }

                if (primitive.attributes.find("WEIGHTS_0") != primitive.attributes.end()) {
                    const tinygltf::Accessor &weightAccessor = model.accessors[primitive.attributes.find("WEIGHTS_0")->second];
                    const tinygltf::BufferView &weightView = model.bufferViews[weightAccessor.bufferView];
                    bufferWeights = reinterpret_cast<const float *>(&(model.buffers[weightView.buffer].data[weightAccessor.byteOffset + weightView.byteOffset]));
                    weightByteStride = weightAccessor.ByteStride(weightView) ? (weightAccessor.ByteStride(weightView) / sizeof(float)) : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC4);
                }

                hasSkin = (bufferJoints && bufferWeights);

                for (size_t v = 0; v < posAccessor.count; v++) {
                    Vertex vert{};
                    vert.pos = glm::vec4(glm::make_vec3(&bufferPos[v * posByteStride]), 1.0f);
                    vert.normal = glm::normalize(glm::vec3(bufferNormals ? glm::make_vec3(&bufferNormals[v * normByteStride]) : glm::vec3(0.0f)));
                    vert.uv0 = bufferTexCoordSet0 ? glm::make_vec2(&bufferTexCoordSet0[v * uv0ByteStride]) : glm::vec3(0.0f);
                    vert.uv1 = bufferTexCoordSet1 ? glm::make_vec2(&bufferTexCoordSet1[v * uv1ByteStride]) : glm::vec3(0.0f);

                    if (hasSkin)
                    {
                        switch (jointComponentType) {
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                            const uint16_t *buf = static_cast<const uint16_t*>(bufferJoints);
                            vert.joint0 = glm::vec4(glm::make_vec4(&buf[v * jointByteStride]));
                            break;
                        }
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
                            const uint8_t *buf = static_cast<const uint8_t*>(bufferJoints);
                            vert.joint0 = glm::vec4(glm::make_vec4(&buf[v * jointByteStride]));
                            break;
                        }
                        default:
                            // Not supported by spec
                            std::cerr << "Joint component type " << jointComponentType << " not supported!" << std::endl;
                            break;
                        }
                    }
                    else {
                        vert.joint0 = glm::vec4(0.0f);
                    }
                    vert.weight0 = hasSkin ? glm::make_vec4(&bufferWeights[v * weightByteStride]) : glm::vec4(0.0f);
                    // Fix for all zero weights
                    if (glm::length(vert.weight0) == 0.0f) {
                        vert.weight0 = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
                    }
                    vertexBuffer.push_back(vert);
                }
            }
            // Indices
            if (hasIndices)
            {
                const tinygltf::Accessor &accessor = model.accessors[primitive.indices > -1 ? primitive.indices : 0];
                const tinygltf::BufferView &bufferView = model.bufferViews[accessor.bufferView];
                const tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];

                indexCount = static_cast<uint32_t>(accessor.count);
                const void *dataPtr = &(buffer.data[accessor.byteOffset + bufferView.byteOffset]);

                switch (accessor.componentType) {
                case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT: {
                    const uint32_t *buf = static_cast<const uint32_t*>(dataPtr);
                    for (size_t index = 0; index < accessor.count; index++) {
                        indexBuffer.push_back(buf[index] + vertexStart);
                    }
                    break;
                }
                case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT: {
                    const uint16_t *buf = static_cast<const uint16_t*>(dataPtr);
                    for (size_t index = 0; index < accessor.count; index++) {
                        indexBuffer.push_back(buf[index] + vertexStart);
                    }
                    break;
                }
                case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE: {
                    const uint8_t *buf = static_cast<const uint8_t*>(dataPtr);
                    for (size_t index = 0; index < accessor.count; index++) {
                        indexBuffer.push_back(buf[index] + vertexStart);
                    }
                    break;
                }
                default:
                    std::cerr << "Index component type " << accessor.componentType << " not supported!" << std::endl;
                    return;
                }
            }
            Primitive *newPrimitive = new Primitive(indexStart, indexCount, vertexCount, primitive.material > -1 ? materials[primitive.material] : materials.back());
            newPrimitive->setBoundingBox(posMin, posMax);
            newMesh->primitives.push_back(newPrimitive);
        }
        // Mesh BB from BBs of primitives
        for (auto p : newMesh->primitives) {
            if (p->bb.valid && !newMesh->bb.valid) {
                newMesh->bb = p->bb;
                newMesh->bb.valid = true;
            }
            newMesh->bb.min = glm::min(newMesh->bb.min, p->bb.min);
            newMesh->bb.max = glm::max(newMesh->bb.max, p->bb.max);
        }
        newNode->mesh = newMesh;
    }
    if (parent) {
        parent->children.push_back(newNode);
    } else {
        nodes.push_back(newNode);
    }
    linearNodes.push_back(newNode);
}

void gltfModel::loadSkins(tinygltf::Model &gltfModel)
{
    for (tinygltf::Skin &source : gltfModel.skins) {
        Skin *newSkin = new Skin{};
        newSkin->name = source.name;

        // Find skeleton root node
        if (source.skeleton > -1) {
            newSkin->skeletonRoot = nodeFromIndex(source.skeleton);
        }

        // Find joint nodes
        for (int jointIndex : source.joints) {
            Node* node = nodeFromIndex(jointIndex);
            if (node) {
                newSkin->joints.push_back(nodeFromIndex(jointIndex));
            }
        }

        // Get inverse bind matrices from buffer
        if (source.inverseBindMatrices > -1) {
            const tinygltf::Accessor &accessor = gltfModel.accessors[source.inverseBindMatrices];
            const tinygltf::BufferView &bufferView = gltfModel.bufferViews[accessor.bufferView];
            const tinygltf::Buffer &buffer = gltfModel.buffers[bufferView.buffer];
            newSkin->inverseBindMatrices.resize(accessor.count);
            memcpy(newSkin->inverseBindMatrices.data(), &buffer.data[accessor.byteOffset + bufferView.byteOffset], accessor.count * sizeof(glm::mat4));
        }

        skins.push_back(newSkin);
    }
}

void gltfModel::loadTextures(VkPhysicalDevice* physicalDevice, VkDevice* device, VkQueue* graphicsQueue, VkCommandPool* commandPool, tinygltf::Model& gltfModel)
{
    for(tinygltf::Texture &tex : gltfModel.textures)
    {
        tinygltf::Image image = gltfModel.images[tex.source];
        textureSampler TextureSampler;
        if (tex.sampler == -1)
        {
            // No sampler specified, use a default one
            TextureSampler.magFilter = VK_FILTER_LINEAR;
            TextureSampler.minFilter = VK_FILTER_LINEAR;
            TextureSampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            TextureSampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            TextureSampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        }
        else
        {
            TextureSampler = textureSamplers[tex.sampler];
        }
        texture Texture;
        Texture.createTextureImage(physicalDevice,device,graphicsQueue,commandPool,image);
        Texture.createTextureImageView(device);
        Texture.createTextureSampler(device,TextureSampler);
        textures.push_back(Texture);
    }
}

VkSamplerAddressMode gltfModel::getVkWrapMode(int32_t wrapMode)
{
    switch (wrapMode) {
    case 10497:
        return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case 33071:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case 33648:
        return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    };
    return VK_SAMPLER_ADDRESS_MODE_REPEAT;            //ADD
}

VkFilter gltfModel::getVkFilterMode(int32_t filterMode)
{
    switch (filterMode) {
    case 9728:
        return VK_FILTER_NEAREST;
    case 9729:
        return VK_FILTER_LINEAR;
    case 9984:
        return VK_FILTER_NEAREST;
    case 9985:
        return VK_FILTER_NEAREST;
    case 9986:
        return VK_FILTER_LINEAR;
    case 9987:
        return VK_FILTER_LINEAR;
    }
    return VK_FILTER_LINEAR;            //ADD
}

void gltfModel::loadTextureSamplers(tinygltf::Model &gltfModel)
{
    for (const tinygltf::Sampler& smpl : gltfModel.samplers) {
        textureSampler sampler{};
        sampler.minFilter = getVkFilterMode(smpl.minFilter);
        sampler.magFilter = getVkFilterMode(smpl.magFilter);
        sampler.addressModeU = getVkWrapMode(smpl.wrapS);
        sampler.addressModeV = getVkWrapMode(smpl.wrapT);
        sampler.addressModeW = sampler.addressModeV;
        textureSamplers.push_back(sampler);
    }
}

void gltfModel::loadMaterials(tinygltf::Model &gltfModel)
{
    for (tinygltf::Material &mat : gltfModel.materials)
    {
        Material material{};
        if (mat.values.find("baseColorTexture") != mat.values.end()) {
            material.baseColorTexture = &textures[mat.values["baseColorTexture"].TextureIndex()];
            material.texCoordSets.baseColor = mat.values["baseColorTexture"].TextureTexCoord();
        }
        if (mat.values.find("metallicRoughnessTexture") != mat.values.end()) {
            material.metallicRoughnessTexture = &textures[mat.values["metallicRoughnessTexture"].TextureIndex()];
            material.texCoordSets.metallicRoughness = mat.values["metallicRoughnessTexture"].TextureTexCoord();
        }
        if (mat.values.find("roughnessFactor") != mat.values.end()) {
            material.roughnessFactor = static_cast<float>(mat.values["roughnessFactor"].Factor());
        }
        if (mat.values.find("metallicFactor") != mat.values.end()) {
            material.metallicFactor = static_cast<float>(mat.values["metallicFactor"].Factor());
        }
        if (mat.values.find("baseColorFactor") != mat.values.end()) {
            material.baseColorFactor = glm::make_vec4(mat.values["baseColorFactor"].ColorFactor().data());
        }
        if (mat.additionalValues.find("normalTexture") != mat.additionalValues.end()) {
            material.normalTexture = &textures[mat.additionalValues["normalTexture"].TextureIndex()];
            material.texCoordSets.normal = mat.additionalValues["normalTexture"].TextureTexCoord();
        }
        if (mat.additionalValues.find("emissiveTexture") != mat.additionalValues.end()) {
            material.emissiveTexture = &textures[mat.additionalValues["emissiveTexture"].TextureIndex()];
            material.texCoordSets.emissive = mat.additionalValues["emissiveTexture"].TextureTexCoord();
        }
        if (mat.additionalValues.find("occlusionTexture") != mat.additionalValues.end()) {
            material.occlusionTexture = &textures[mat.additionalValues["occlusionTexture"].TextureIndex()];
            material.texCoordSets.occlusion = mat.additionalValues["occlusionTexture"].TextureTexCoord();
        }
        if (mat.additionalValues.find("alphaMode") != mat.additionalValues.end()) {
            tinygltf::Parameter param = mat.additionalValues["alphaMode"];
            if (param.string_value == "BLEND") {
                material.alphaMode = Material::ALPHAMODE_BLEND;
            }
            if (param.string_value == "MASK") {
                material.alphaCutoff = 0.5f;
                material.alphaMode = Material::ALPHAMODE_MASK;
            }
        }
        if (mat.additionalValues.find("alphaCutoff") != mat.additionalValues.end()) {
            material.alphaCutoff = static_cast<float>(mat.additionalValues["alphaCutoff"].Factor());
        }
        if (mat.additionalValues.find("emissiveFactor") != mat.additionalValues.end()) {
            material.emissiveFactor = glm::vec4(glm::make_vec3(mat.additionalValues["emissiveFactor"].ColorFactor().data()), 1.0);
            material.emissiveFactor = glm::vec4(0.0f);
        }

        // Extensions
        // @TODO: Find out if there is a nicer way of reading these properties with recent tinygltf headers
        if (mat.extensions.find("KHR_materials_pbrSpecularGlossiness") != mat.extensions.end()) {
            auto ext = mat.extensions.find("KHR_materials_pbrSpecularGlossiness");
            if (ext->second.Has("specularGlossinessTexture")) {
                auto index = ext->second.Get("specularGlossinessTexture").Get("index");
                material.extension.specularGlossinessTexture = &textures[index.Get<int>()];
                auto texCoordSet = ext->second.Get("specularGlossinessTexture").Get("texCoord");
                material.texCoordSets.specularGlossiness = texCoordSet.Get<int>();
                material.pbrWorkflows.specularGlossiness = true;
            }
            if (ext->second.Has("diffuseTexture")) {
                auto index = ext->second.Get("diffuseTexture").Get("index");
                material.extension.diffuseTexture = &textures[index.Get<int>()];
            }
            if (ext->second.Has("diffuseFactor")) {
                auto factor = ext->second.Get("diffuseFactor");
                for (uint32_t i = 0; i < factor.ArrayLen(); i++) {
                    auto val = factor.Get(i);
                    material.extension.diffuseFactor[i] = val.IsNumber() ? (float)val.Get<double>() : (float)val.Get<int>();
                }
            }
            if (ext->second.Has("specularFactor")) {
                auto factor = ext->second.Get("specularFactor");
                for (uint32_t i = 0; i < factor.ArrayLen(); i++) {
                    auto val = factor.Get(i);
                    material.extension.specularFactor[i] = val.IsNumber() ? (float)val.Get<double>() : (float)val.Get<int>();
                }
            }
        }

        materials.push_back(material);
    }
    // Push a default material at the end of the list for meshes with no material assigned
    materials.push_back(Material());
}

void gltfModel::loadAnimations(tinygltf::Model &gltfModel)
{
    for (tinygltf::Animation &anim : gltfModel.animations) {
        Animation animation{};
        animation.name = anim.name;
        if (anim.name.empty()) {
            animation.name = std::to_string(animations.size());
        }

        // Samplers
        for (auto &samp : anim.samplers) {
            AnimationSampler sampler{};

            if (samp.interpolation == "LINEAR") {
                sampler.interpolation = AnimationSampler::InterpolationType::LINEAR;
            }
            if (samp.interpolation == "STEP") {
                sampler.interpolation = AnimationSampler::InterpolationType::STEP;
            }
            if (samp.interpolation == "CUBICSPLINE") {
                sampler.interpolation = AnimationSampler::InterpolationType::CUBICSPLINE;
            }

            // Read sampler input time values
            {
                const tinygltf::Accessor &accessor = gltfModel.accessors[samp.input];
                const tinygltf::BufferView &bufferView = gltfModel.bufferViews[accessor.bufferView];
                const tinygltf::Buffer &buffer = gltfModel.buffers[bufferView.buffer];

                assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);

                const void *dataPtr = &buffer.data[accessor.byteOffset + bufferView.byteOffset];
                const float *buf = static_cast<const float*>(dataPtr);
                for (size_t index = 0; index < accessor.count; index++) {
                    sampler.inputs.push_back(buf[index]);
                }

                for (auto input : sampler.inputs) {
                    if (input < animation.start) {
                        animation.start = input;
                    };
                    if (input > animation.end) {
                        animation.end = input;
                    }
                }
            }

            // Read sampler output T/R/S values
            {
                const tinygltf::Accessor &accessor = gltfModel.accessors[samp.output];
                const tinygltf::BufferView &bufferView = gltfModel.bufferViews[accessor.bufferView];
                const tinygltf::Buffer &buffer = gltfModel.buffers[bufferView.buffer];

                assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);

                const void *dataPtr = &buffer.data[accessor.byteOffset + bufferView.byteOffset];

                switch (accessor.type) {
                    case TINYGLTF_TYPE_VEC3: {
                        const glm::vec3 *buf = static_cast<const glm::vec3*>(dataPtr);
                        for (size_t index = 0; index < accessor.count; index++) {
                            sampler.outputsVec4.push_back(glm::vec4(buf[index], 0.0f));
                        }
                        break;
                    }
                    case TINYGLTF_TYPE_VEC4: {
                        const glm::vec4 *buf = static_cast<const glm::vec4*>(dataPtr);
                        for (size_t index = 0; index < accessor.count; index++) {
                            sampler.outputsVec4.push_back(buf[index]);
                        }
                        break;
                    }
                    default: {
                        std::cout << "unknown type" << std::endl;
                        break;
                    }
                }
            }

            animation.samplers.push_back(sampler);
        }

        // Channels
        for (auto &source: anim.channels)
        {
            AnimationChannel channel{};

            if (source.target_path == "rotation") {
                channel.path = AnimationChannel::PathType::ROTATION;
            }
            if (source.target_path == "translation") {
                channel.path = AnimationChannel::PathType::TRANSLATION;
            }
            if (source.target_path == "scale") {
                channel.path = AnimationChannel::PathType::SCALE;
            }
            if (source.target_path == "weights") {
                std::cout << "weights not yet supported, skipping channel" << std::endl;
                continue;
            }
            channel.samplerIndex = source.sampler;
            channel.node = nodeFromIndex(source.target_node);
            if (!channel.node) {
                continue;
            }

            animation.channels.push_back(channel);
        }

        animations.push_back(animation);
    }
}

void gltfModel::loadFromFile(VkPhysicalDevice* physicalDevice, VkDevice* device, VkQueue* graphicsQueue, VkCommandPool* commandPool, float scale)
{
    tinygltf::Model gltfModel;
    tinygltf::TinyGLTF gltfContext;
    std::string error;
    std::string warning;

    bool binary = false;
    size_t extpos = filename.rfind('.', filename.length());
    if (extpos != std::string::npos) {
        binary = (filename.substr(extpos + 1, filename.length() - extpos) == "glb");
    }

    bool fileLoaded = binary ? gltfContext.LoadBinaryFromFile(&gltfModel, &error, &warning, filename.c_str()) : gltfContext.LoadASCIIFromFile(&gltfModel, &error, &warning, filename.c_str());

    std::vector<uint32_t> indexBuffer;
    std::vector<Vertex> vertexBuffer;

    if (fileLoaded)
    {
        loadTextureSamplers(gltfModel);
        loadTextures(physicalDevice,device,graphicsQueue,commandPool,gltfModel);
        loadMaterials(gltfModel);

        // TODO: scene handling with no default scene
        const tinygltf::Scene &scene = gltfModel.scenes[gltfModel.defaultScene > -1 ? gltfModel.defaultScene : 0];
        for (size_t i = 0; i < scene.nodes.size(); i++) {
            const tinygltf::Node node = gltfModel.nodes[scene.nodes[i]];
            loadNode(physicalDevice, device, nullptr, node, scene.nodes[i], gltfModel, indexBuffer, vertexBuffer, scale);
        }
        if (gltfModel.animations.size() > 0) {
            loadAnimations(gltfModel);
        }
        loadSkins(gltfModel);

        for (auto node : linearNodes) {
            // Assign skins
            if (node->skinIndex > -1) {
                node->skin = skins[node->skinIndex];
            }
            // Initial pose
            if (node->mesh) {
                node->update();
            }
        }
    }
    else {
        // TODO: throw
        std::cerr << "Could not load gltf file: " << error << std::endl;
        return;
    }

    calculateTangent(vertexBuffer, indexBuffer);

    extensions = gltfModel.extensionsUsed;

    size_t vertexBufferSize = vertexBuffer.size() * sizeof(Vertex);
    size_t indexBufferSize = indexBuffer.size() * sizeof(uint32_t);
    indices.count = static_cast<uint32_t>(indexBuffer.size());

    assert(vertexBufferSize > 0);

    struct StagingBuffer {
        VkBuffer buffer;
        VkDeviceMemory memory;
    } vertexStaging, indexStaging;

    // Create staging buffers
    // Vertex data
    createBuffer(physicalDevice,device,vertexBufferSize,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,vertexStaging.buffer,vertexStaging.memory);
    // Index data
    createBuffer(physicalDevice,device,indexBufferSize,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,indexStaging.buffer,indexStaging.memory);

    void* data;
    vkMapMemory(*device, vertexStaging.memory, 0, vertexBufferSize, 0, &data);
        memcpy(data, vertexBuffer.data(), (size_t) vertexBufferSize);
    vkUnmapMemory(*device, vertexStaging.memory);

    void* indexdata;
    vkMapMemory(*device, indexStaging.memory, 0, indexBufferSize, 0, &indexdata);
        memcpy(indexdata, indexBuffer.data(), (size_t) indexBufferSize);
    vkUnmapMemory(*device, indexStaging.memory);

    // Create device local buffers
    // Vertex buffer
    createBuffer(physicalDevice,device,vertexBufferSize,VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,vertices.buffer,vertices.memory);
    // Index buffer
    createBuffer(physicalDevice,device,indexBufferSize,VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,indices.buffer,indices.memory);

    // Copy from staging buffers
    copyBuffer(device, graphicsQueue, commandPool, vertexStaging.buffer, vertices.buffer, vertexBufferSize);
    copyBuffer(device, graphicsQueue, commandPool, indexStaging.buffer, indices.buffer, indexBufferSize);

    vkDestroyBuffer(*device, vertexStaging.buffer, nullptr);
    vkFreeMemory(*device, vertexStaging.memory, nullptr);
    if (indexBufferSize > 0) {
        vkDestroyBuffer(*device, indexStaging.buffer, nullptr);
        vkFreeMemory(*device, indexStaging.memory, nullptr);
    }

    getSceneDimensions();
}

void gltfModel::calculateBoundingBox(Node *node, Node *parent)
{
    BoundingBox parentBvh = parent ? parent->bvh : BoundingBox(dimensions.min, dimensions.max);

    if (node->mesh) {
        if (node->mesh->bb.valid) {
            node->aabb = node->mesh->bb.getAABB(node->getMatrix());
            if (node->children.size() == 0) {
                node->bvh.min = node->aabb.min;
                node->bvh.max = node->aabb.max;
                node->bvh.valid = true;
            }
        }
    }

    parentBvh.min = glm::min(parentBvh.min, node->bvh.min);
    parentBvh.max = glm::min(parentBvh.max, node->bvh.max);

    for (auto &child : node->children) {
        calculateBoundingBox(child, node);
    }
}

void gltfModel::getSceneDimensions()
{
    // Calculate binary volume hierarchy for all nodes in the scene
    for (auto node : linearNodes) {
        calculateBoundingBox(node, nullptr);
    }

    dimensions.min = glm::vec3(FLT_MAX);
    dimensions.max = glm::vec3(-FLT_MAX);

    for (auto node : linearNodes) {
        if (node->bvh.valid) {
            dimensions.min = glm::min(dimensions.min, node->bvh.min);
            dimensions.max = glm::max(dimensions.max, node->bvh.max);
        }
    }

    // Calculate scene aabb
    aabb = glm::scale(glm::mat4(1.0f), glm::vec3(dimensions.max[0] - dimensions.min[0], dimensions.max[1] - dimensions.min[1], dimensions.max[2] - dimensions.min[2]));
    aabb[3][0] = dimensions.min[0];
    aabb[3][1] = dimensions.min[1];
    aabb[3][2] = dimensions.min[2];
}

void gltfModel::updateAnimation(uint32_t index, float time)
{
    if (animations.empty()) {
        std::cout << ".glTF does not contain animation." << std::endl;
        return;
    }
    if (index > static_cast<uint32_t>(animations.size()) - 1) {
        std::cout << "No animation with index " << index << std::endl;
        return;
    }
    Animation &animation = animations[index];

    bool updated = false;
    for (auto& channel : animation.channels) {
        AnimationSampler &sampler = animation.samplers[channel.samplerIndex];
        if (sampler.inputs.size() > sampler.outputsVec4.size()) {
            continue;
        }

        for (size_t i = 0; i < sampler.inputs.size() - 1; i++) {
            if ((time >= sampler.inputs[i]) && (time <= sampler.inputs[i + 1])) {
                float u = std::max(0.0f, time - sampler.inputs[i]) / (sampler.inputs[i + 1] - sampler.inputs[i]);
                if (u <= 1.0f) {
                    switch (channel.path) {
                    case AnimationChannel::PathType::TRANSLATION: {
                        glm::vec4 trans = glm::mix(sampler.outputsVec4[i], sampler.outputsVec4[i + 1], u);
                        channel.node->translation = glm::vec3(trans);
                        break;
                    }
                    case AnimationChannel::PathType::SCALE: {
                        glm::vec4 trans = glm::mix(sampler.outputsVec4[i], sampler.outputsVec4[i + 1], u);
                        channel.node->scale = glm::vec3(trans);
                        break;
                    }
                    case AnimationChannel::PathType::ROTATION: {
                        glm::quat q1;
                        q1.x = sampler.outputsVec4[i].x;
                        q1.y = sampler.outputsVec4[i].y;
                        q1.z = sampler.outputsVec4[i].z;
                        q1.w = sampler.outputsVec4[i].w;
                        glm::quat q2;
                        q2.x = sampler.outputsVec4[i + 1].x;
                        q2.y = sampler.outputsVec4[i + 1].y;
                        q2.z = sampler.outputsVec4[i + 1].z;
                        q2.w = sampler.outputsVec4[i + 1].w;
                        channel.node->rotation = glm::normalize(glm::slerp(q1, q2, u));
                        break;
                    }
                    }
                    updated = true;
                }
            }
        }
    }
    if (updated) {
        for (auto &node : nodes) {
            node->update();
        }
    }
}

void gltfModel::changeAnimation(uint32_t oldIndex, uint32_t newIndex, float startTime, float time, float changeAnimationTime)
{
    Animation &animationOld = animations[oldIndex];
    Animation &animationNew = animations[newIndex];

    bool updated = false;
    for (auto& channel : animationOld.channels) {
        AnimationSampler &samplerOld = animationOld.samplers[channel.samplerIndex];
        AnimationSampler &samplerNew = animationNew.samplers[channel.samplerIndex];
        if (samplerOld.inputs.size() > samplerOld.outputsVec4.size())
            continue;

        for (size_t i = 0; i < samplerOld.inputs.size(); i++) {
            if ((startTime >= samplerOld.inputs[i]) && (time <= samplerOld.inputs[i]+changeAnimationTime)) {
                float u = std::max(0.0f, time - startTime) / changeAnimationTime;
                if (u <= 1.0f) {
                    switch (channel.path) {
                        case AnimationChannel::PathType::TRANSLATION: {
                            glm::vec4 trans = glm::mix(samplerOld.outputsVec4[i], samplerNew.outputsVec4[0], u);
                            channel.node->translation = glm::vec3(trans);
                            break;
                        }
                        case AnimationChannel::PathType::SCALE: {
                            glm::vec4 trans = glm::mix(samplerOld.outputsVec4[i], samplerNew.outputsVec4[0], u);
                            channel.node->scale = glm::vec3(trans);
                            break;
                        }
                        case AnimationChannel::PathType::ROTATION: {
                            glm::quat q1;
                            q1.x = samplerOld.outputsVec4[i].x;
                            q1.y = samplerOld.outputsVec4[i].y;
                            q1.z = samplerOld.outputsVec4[i].z;
                            q1.w = samplerOld.outputsVec4[i].w;
                            glm::quat q2;
                            q2.x = samplerNew.outputsVec4[0].x;
                            q2.y = samplerNew.outputsVec4[0].y;
                            q2.z = samplerNew.outputsVec4[0].z;
                            q2.w = samplerNew.outputsVec4[0].w;
                            channel.node->rotation = glm::normalize(glm::slerp(q1, q2, u));
                            break;
                        }
                    }
                    updated = true;
                }
            }
        }
    }
    if (updated) {
        for (auto &node : nodes) {
            node->update();
        }
    }
}

Node* gltfModel::findNode(Node *parent, uint32_t index)
{
    Node* nodeFound = nullptr;
    if (parent->index == index) {
        return parent;
    }
    for (auto& child : parent->children) {
        nodeFound = findNode(child, index);
        if (nodeFound) {
            break;
        }
    }
    return nodeFound;
}

Node* gltfModel::nodeFromIndex(uint32_t index)
{
    Node* nodeFound = nullptr;
    for (auto &node : nodes) {
        nodeFound = findNode(node, index);
        if (nodeFound) {
            break;
        }
    }
    return nodeFound;
}

void gltfModel::calculateTangent(std::vector<Vertex>& vertexBuffer, std::vector<uint32_t>& indexBuffer)
{
    for (auto& node : nodes) {
        calculateNodeTangent(node, vertexBuffer, indexBuffer);
    }
}
void gltfModel::calculateNodeTangent(Node* node, std::vector<Vertex>& vertexBuffer, std::vector<uint32_t>& indexBuffer)
{
    if (node->mesh) {
        for (Primitive *primitive : node->mesh->primitives) {
            for(uint32_t i = primitive->firstIndex; i<primitive->firstIndex+primitive->indexCount; i += 3){

                glm::vec3 &v1   = vertexBuffer[indexBuffer[i+0]].pos;
                glm::vec3 &v2   = vertexBuffer[indexBuffer[i+1]].pos;
                glm::vec3 &v3   = vertexBuffer[indexBuffer[i+2]].pos;

                glm::vec2 &uv1  = vertexBuffer[indexBuffer[i+0]].uv0;
                glm::vec2 &uv2  = vertexBuffer[indexBuffer[i+1]].uv0;
                glm::vec2 &uv3  = vertexBuffer[indexBuffer[i+2]].uv0;

                glm::vec3 dv1   = v2 - v1;
                glm::vec3 dv2   = v3 - v1;

                glm::vec2 duv1  = uv2 - uv1;
                glm::vec2 duv2  = uv3 - uv1;

                float det = 1.0f/(duv1.x*duv2.y - duv1.y*duv2.x);
                glm::vec3 tangent = det*(duv2.y*dv1-duv1.y*dv2);
                tangent = glm::normalize(tangent);
                glm::vec3 bitangent = det*(duv1.x*dv2-duv2.x*dv1);
                bitangent = glm::normalize(bitangent);

                if(dot(glm::cross(tangent,bitangent),vertexBuffer[indexBuffer[i+0]].normal)<0.0f)
                    tangent = -tangent;

                vertexBuffer[indexBuffer[i+0]].tangent      = glm::normalize(tangent - vertexBuffer[indexBuffer[i+0]].normal * glm::dot(vertexBuffer[indexBuffer[i+0]].normal, tangent));
                vertexBuffer[indexBuffer[i+1]].tangent      = glm::normalize(tangent - vertexBuffer[indexBuffer[i+1]].normal * glm::dot(vertexBuffer[indexBuffer[i+1]].normal, tangent));
                vertexBuffer[indexBuffer[i+2]].tangent      = glm::normalize(tangent - vertexBuffer[indexBuffer[i+2]].normal * glm::dot(vertexBuffer[indexBuffer[i+2]].normal, tangent));

                vertexBuffer[indexBuffer[i+0]].bitangent    = glm::normalize(glm::cross(vertexBuffer[indexBuffer[i+0]].normal,vertexBuffer[indexBuffer[i+0]].tangent));
                vertexBuffer[indexBuffer[i+1]].bitangent    = glm::normalize(glm::cross(vertexBuffer[indexBuffer[i+1]].normal,vertexBuffer[indexBuffer[i+1]].tangent));
                vertexBuffer[indexBuffer[i+2]].bitangent    = glm::normalize(glm::cross(vertexBuffer[indexBuffer[i+2]].normal,vertexBuffer[indexBuffer[i+2]].tangent));
            }
        }
    }
    for (auto& child : node->children) {
        calculateNodeTangent(child, vertexBuffer,indexBuffer);
    }
}

VkVertexInputBindingDescription gltfModel::Vertex::getBloomBindingDescription()
{
    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return bindingDescription;
}

std::array<VkVertexInputAttributeDescription, 5> gltfModel::Vertex::getBloomAttributeDescriptions()
{
    std::array<VkVertexInputAttributeDescription, 5> attributeDescriptions{};

    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex, pos);

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, uv0);

    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[2].offset = offsetof(Vertex, uv1);

    attributeDescriptions[3].binding = 0;
    attributeDescriptions[3].location = 3;
    attributeDescriptions[3].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributeDescriptions[3].offset = offsetof(Vertex, joint0);

    attributeDescriptions[4].binding = 0;
    attributeDescriptions[4].location = 4;
    attributeDescriptions[4].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributeDescriptions[4].offset = offsetof(Vertex, weight0);

    return attributeDescriptions;
}

VkVertexInputBindingDescription gltfModel::Vertex::getStencilBindingDescription()
{
    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return bindingDescription;
}

std::array<VkVertexInputAttributeDescription, 8> gltfModel::Vertex::getStencilAttributeDescriptions()
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

VkVertexInputBindingDescription gltfModel::Vertex::getShadowBindingDescription()
{
    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return bindingDescription;
}

std::array<VkVertexInputAttributeDescription, 3> gltfModel::Vertex::getShadowAttributeDescriptions()
{
    std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

    attributeDescriptions[0].binding = 0;                           //привязка к которой буфер привязан и из которого этот атрибут берёт данные
    attributeDescriptions[0].location = 0;                          //положение которое используется для обращения к атрибуту из вершинного шейдера
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;   //формат вершинных данных
    attributeDescriptions[0].offset = offsetof(Vertex, pos);        //смещение внутри каждой структуры

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, joint0);

    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributeDescriptions[2].offset = offsetof(Vertex, weight0);

    return attributeDescriptions;
}

void gltfModel::createDescriptorPool(VkDevice* device)
{
    uint32_t imageSamplerCount = 0;
    uint32_t materialCount = 0;
    uint32_t meshCount = 0;
    for (auto &material : materials)
    {
        static_cast<void>(material);
        imageSamplerCount += 5;
        materialCount++;
    }

    for (auto node : linearNodes){
        if(node->mesh){
            meshCount++;
        }
    }

    size_t index = 0;
    std::vector<VkDescriptorPoolSize> DescriptorPoolSizes(2);
        DescriptorPoolSizes.at(index).type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        DescriptorPoolSizes.at(index).descriptorCount = meshCount;
    index++;
        DescriptorPoolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        DescriptorPoolSizes.at(index).descriptorCount = imageSamplerCount;
    index++;

    //Мы будем выделять один из этих дескрипторов для каждого кадра. На эту структуру размера пула ссылается главный VkDescriptorPoolCreateInfo:
    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(DescriptorPoolSizes.size());
        poolInfo.pPoolSizes = DescriptorPoolSizes.data();
        poolInfo.maxSets = meshCount+imageSamplerCount;
    if (vkCreateDescriptorPool(*device, &poolInfo, nullptr, &DescriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create object descriptor pool!");
}

void gltfModel::createDescriptorSet(VkDevice* device, texture* emptyTexture)
{
    createMaterialDescriptorSetLayout(device,&materialDescriptorSetLayout);
    createNodeDescriptorSetLayout(device,&nodeDescriptorSetLayout);

    for (auto node : linearNodes){
        if(node->mesh){
            createNodeDescriptorSet(device, node);
        }
    }

    for (auto &material : materials){
        createMaterialDescriptorSet(device, &material, emptyTexture);
    }
}

void gltfModel::createNodeDescriptorSet(VkDevice* device, Node* node)
{
    if (node->mesh)
    {
        VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
            descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            descriptorSetAllocInfo.descriptorPool = DescriptorPool;
            descriptorSetAllocInfo.pSetLayouts = &nodeDescriptorSetLayout;
            descriptorSetAllocInfo.descriptorSetCount = 1;
        if (vkAllocateDescriptorSets(*device, &descriptorSetAllocInfo, &node->mesh->uniformBuffer.descriptorSet) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate object descriptor sets!");

        VkWriteDescriptorSet writeDescriptorSet{};
            writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writeDescriptorSet.descriptorCount = 1;
            writeDescriptorSet.dstSet = node->mesh->uniformBuffer.descriptorSet;
            writeDescriptorSet.dstBinding = 0;
            writeDescriptorSet.pBufferInfo = &node->mesh->uniformBuffer.descriptor;
        vkUpdateDescriptorSets(*device, 1, &writeDescriptorSet, 0, nullptr);
    }
    for (auto& child : node->children)
        createNodeDescriptorSet(device,child);
}

void gltfModel::createMaterialDescriptorSet(VkDevice* device, Material* material, texture* emptyTexture)
{
    std::vector<VkDescriptorSetLayout> layouts(1, materialDescriptorSetLayout);
    VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
        descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocInfo.descriptorPool = DescriptorPool;
        descriptorSetAllocInfo.pSetLayouts = layouts.data();
        descriptorSetAllocInfo.descriptorSetCount = 1;
    if (vkAllocateDescriptorSets(*device, &descriptorSetAllocInfo, &material->descriptorSet) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate object descriptor sets!");

    VkDescriptorImageInfo baseColorTextureInfo;
    baseColorTextureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (material->pbrWorkflows.metallicRoughness)
    {
        baseColorTextureInfo.imageView   = material->baseColorTexture ? *material->baseColorTexture->getTextureImageView() : *emptyTexture->getTextureImageView();
        baseColorTextureInfo.sampler     = material->baseColorTexture ? *material->baseColorTexture->getTextureSampler()   : *emptyTexture->getTextureSampler();
    }
    if(material->pbrWorkflows.specularGlossiness)
    {
        baseColorTextureInfo.imageView   = material->extension.diffuseTexture ? *material->extension.diffuseTexture->getTextureImageView() : *emptyTexture->getTextureImageView();
        baseColorTextureInfo.sampler     = material->extension.diffuseTexture ? *material->extension.diffuseTexture->getTextureSampler() : *emptyTexture->getTextureSampler();
    }

    VkDescriptorImageInfo metallicRoughnessTextureInfo;
    metallicRoughnessTextureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (material->pbrWorkflows.metallicRoughness)
    {
        metallicRoughnessTextureInfo.imageView   = material->metallicRoughnessTexture ? *material->metallicRoughnessTexture->getTextureImageView() : *emptyTexture->getTextureImageView();
        metallicRoughnessTextureInfo.sampler     = material->metallicRoughnessTexture ? *material->metallicRoughnessTexture->getTextureSampler() : *emptyTexture->getTextureSampler();
    }
    if (material->pbrWorkflows.specularGlossiness)
    {
        metallicRoughnessTextureInfo.imageView   = material->extension.specularGlossinessTexture ? *material->extension.specularGlossinessTexture->getTextureImageView() : *emptyTexture->getTextureImageView();
        metallicRoughnessTextureInfo.sampler     = material->extension.specularGlossinessTexture ? *material->extension.specularGlossinessTexture->getTextureSampler() : *emptyTexture->getTextureSampler();
    }

    VkDescriptorImageInfo normalTextureInfo;
    normalTextureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    normalTextureInfo.imageView   = material->normalTexture ? *material->normalTexture->getTextureImageView() : *emptyTexture->getTextureImageView();
    normalTextureInfo.sampler     = material->normalTexture ? *material->normalTexture->getTextureSampler() : *emptyTexture->getTextureSampler();

    VkDescriptorImageInfo occlusionTextureInfo;
    occlusionTextureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    occlusionTextureInfo.imageView   = material->occlusionTexture ? *material->occlusionTexture->getTextureImageView() : *emptyTexture->getTextureImageView();
    occlusionTextureInfo.sampler     = material->occlusionTexture ? *material->occlusionTexture->getTextureSampler() : *emptyTexture->getTextureSampler();

    VkDescriptorImageInfo emissiveTextureInfo;
    emissiveTextureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    emissiveTextureInfo.imageView   = material->emissiveTexture ? *material->emissiveTexture->getTextureImageView() : *emptyTexture->getTextureImageView();
    emissiveTextureInfo.sampler     = material->emissiveTexture ? *material->emissiveTexture->getTextureSampler() : *emptyTexture->getTextureSampler();

    std::array<VkDescriptorImageInfo, 5> descriptorImageInfos = {baseColorTextureInfo,metallicRoughnessTextureInfo,normalTextureInfo,occlusionTextureInfo,emissiveTextureInfo};
    std::array<VkWriteDescriptorSet, 5> descriptorWrites{};

    for(size_t i=0;i<5;i++)
    {
        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].dstSet = material->descriptorSet;
        descriptorWrites[i].dstBinding = i;
        descriptorWrites[i].dstArrayElement = 0;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pImageInfo = &descriptorImageInfos[i];
    }
    vkUpdateDescriptorSets(*device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);

}
