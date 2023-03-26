#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define STBI_MSC_SECURE_CRT

#include "gltfmodel.h"
#include "../utils/operations.h"

#include <iostream>
#include <fstream>
#include <cstring>

namespace {
    glm::mat4 localMatrix(Node* node){
        return glm::translate(glm::mat4(1.0f), node->translation) * glm::mat4(node->rotation) * glm::scale(glm::mat4(1.0f), node->scale) * node->matrix;
    }

    glm::mat4 getMatrix(Node* node) {
        return (node->parent ? getMatrix(node->parent) : glm::mat4(1.0f)) * localMatrix(node);
    }

    VkSamplerAddressMode getVkWrapMode(int32_t wrapMode){
        switch (wrapMode) {
        case 10497:
            return VK_SAMPLER_ADDRESS_MODE_REPEAT;
        case 33071:
            return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        case 33648:
            return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        };
        return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    }

    VkFilter getVkFilterMode(int32_t filterMode){
        switch (filterMode) {
        case 9728:
        case 9984:
        case 9985:
            return VK_FILTER_NEAREST;
        case 9729:
        case 9986:
        case 9987:
            return VK_FILTER_LINEAR;
        }
        return VK_FILTER_LINEAR;
    }

    void calculateNodeTangent(Node* node, std::vector<model::Vertex>& vertexBuffer, std::vector<uint32_t>& indexBuffer){
        if (node->mesh) {
            for (Mesh::Primitive *primitive : node->mesh->primitives) {
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

    Node* findNodeFromIndex(Node* node, uint32_t index){
        Node* nodeFound = nullptr;
        if (node->index == index) {
            nodeFound = node;
        } else {
            for (auto& child : node->children) {
                nodeFound = findNodeFromIndex(child, index);
                if (nodeFound) break;
            }
        }
        return nodeFound;
    }

    Node* nodeFromIndex(uint32_t index, const std::vector<Node*>& nodes)
    {
        Node* nodeFound = nullptr;
        for (const auto &node : nodes) {
            nodeFound = findNodeFromIndex(node, index);
            if (nodeFound) break;
        }
        return nodeFound;
    }

    bool isBinary(const std::string& filename){
        size_t extpos = filename.rfind('.', filename.length());
        return (extpos != std::string::npos) && (filename.substr(extpos + 1, filename.length() - extpos) == "glb");
    }

    void createNodeDescriptorSet(VkDevice device, Node* node, VkDescriptorPool descriptorPool, VkDescriptorSetLayout descriptorSetLayout)
    {
        if (node->mesh){
            VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
                descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                descriptorSetAllocInfo.descriptorPool = descriptorPool;
                descriptorSetAllocInfo.pSetLayouts = &descriptorSetLayout;
                descriptorSetAllocInfo.descriptorSetCount = 1;
            vkAllocateDescriptorSets(device, &descriptorSetAllocInfo, &node->mesh->uniformBuffer.descriptorSet);

            VkDescriptorBufferInfo bufferInfo{
                node->mesh->uniformBuffer.instance,
                0,
                sizeof(Mesh::uniformBlock)
            };

            VkWriteDescriptorSet writeDescriptorSet{};
                writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                writeDescriptorSet.descriptorCount = 1;
                writeDescriptorSet.dstSet = node->mesh->uniformBuffer.descriptorSet;
                writeDescriptorSet.dstBinding = 0;
                writeDescriptorSet.pBufferInfo = &bufferInfo;
            vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
        }
        for (auto& child : node->children){
            createNodeDescriptorSet(device,child,descriptorPool,descriptorSetLayout);
        }
    }

    void createMaterialDescriptorSet(VkDevice device, Material* material, texture* emptyTexture, VkDescriptorPool descriptorPool, VkDescriptorSetLayout descriptorSetLayout)
    {
        VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
            descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            descriptorSetAllocInfo.descriptorPool = descriptorPool;
            descriptorSetAllocInfo.pSetLayouts = &descriptorSetLayout;
            descriptorSetAllocInfo.descriptorSetCount = 1;
        vkAllocateDescriptorSets(device, &descriptorSetAllocInfo, &material->descriptorSet);

        VkDescriptorImageInfo baseColorTextureInfo;
        baseColorTextureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        if (material->pbrWorkflows.metallicRoughness){
            baseColorTextureInfo.imageView   = material->baseColorTexture ? *material->baseColorTexture->getTextureImageView() : *emptyTexture->getTextureImageView();
            baseColorTextureInfo.sampler     = material->baseColorTexture ? *material->baseColorTexture->getTextureSampler()   : *emptyTexture->getTextureSampler();
        }
        if(material->pbrWorkflows.specularGlossiness){
            baseColorTextureInfo.imageView   = material->extension.diffuseTexture ? *material->extension.diffuseTexture->getTextureImageView() : *emptyTexture->getTextureImageView();
            baseColorTextureInfo.sampler     = material->extension.diffuseTexture ? *material->extension.diffuseTexture->getTextureSampler() : *emptyTexture->getTextureSampler();
        }

        VkDescriptorImageInfo metallicRoughnessTextureInfo;
        metallicRoughnessTextureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        if (material->pbrWorkflows.metallicRoughness){
            metallicRoughnessTextureInfo.imageView   = material->metallicRoughnessTexture ? *material->metallicRoughnessTexture->getTextureImageView() : *emptyTexture->getTextureImageView();
            metallicRoughnessTextureInfo.sampler     = material->metallicRoughnessTexture ? *material->metallicRoughnessTexture->getTextureSampler() : *emptyTexture->getTextureSampler();
        }
        if (material->pbrWorkflows.specularGlossiness){
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

        std::vector<VkDescriptorImageInfo> descriptorImageInfos = {
            baseColorTextureInfo,
            metallicRoughnessTextureInfo,
            normalTextureInfo,
            occlusionTextureInfo,
            emissiveTextureInfo
        };

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        for(const auto& info: descriptorImageInfos){
            descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = material->descriptorSet;
            descriptorWrites.back().dstBinding = descriptorWrites.size() - 1;
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &info;
        }
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

BoundingBox::BoundingBox(glm::vec3 min, glm::vec3 max)
    : min(min), max(max), valid(true)
{};

BoundingBox BoundingBox::getAABB(glm::mat4 m) const {
    return BoundingBox(
        glm::min(glm::vec3(m[0]) * this->min.x, glm::vec3(m[0]) * this->max.x)
        + glm::min(glm::vec3(m[1]) * this->min.y, glm::vec3(m[1]) * this->max.y)
        + glm::min(glm::vec3(m[2]) * this->min.z, glm::vec3(m[2]) * this->max.z),
        glm::max(glm::vec3(m[0]) * this->min.x, glm::vec3(m[0]) * this->max.x)
        + glm::max(glm::vec3(m[1]) * this->min.y, glm::vec3(m[1]) * this->max.y)
        + glm::max(glm::vec3(m[2]) * this->min.z, glm::vec3(m[2]) * this->max.z));
}

Mesh::Primitive::Primitive(uint32_t firstIndex, uint32_t indexCount, uint32_t vertexCount, Material &material)
    : firstIndex(firstIndex), indexCount(indexCount), vertexCount(vertexCount), material(material), hasIndices(indexCount > 0)
{}

Mesh::Mesh(VkPhysicalDevice physicalDevice, VkDevice device, glm::mat4 matrix)
{
    this->uniformBlock.matrix = matrix;
    Buffer::create( physicalDevice,
                    device,
                    sizeof(uniformBlock),
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    &uniformBuffer.instance,
                    &uniformBuffer.memory);
    vkMapMemory(device, uniformBuffer.memory, 0, sizeof(uniformBlock), 0, &uniformBuffer.map);
};

void Mesh::destroy(VkDevice device){
    uniformBuffer.destroy(device);
    for (Primitive* p : primitives){
        delete p;
    }
}

void Node::update() {
    if (size_t numJoints = skin ? std::min((uint32_t)skin->joints.size(), MAX_NUM_JOINTS) : 0; mesh){
        mesh->uniformBlock.matrix = getMatrix(this);
        for (size_t i = 0; i < numJoints; i++) {
            mesh->uniformBlock.jointMatrix[i] = glm::inverse(mesh->uniformBlock.matrix) * getMatrix(skin->joints[i]) * skin->inverseBindMatrices[i];
        }
        mesh->uniformBlock.jointcount = static_cast<float>(numJoints);
        std::memcpy(mesh->uniformBuffer.map, &mesh->uniformBlock, sizeof(mesh->uniformBlock));
    }
    for (auto& child : children){
        child->update();
    }
}

size_t Node::meshCount() const {
    return std::accumulate(children.begin(), children.end(), mesh ? 1 : 0, [](const size_t& count, Node* node){
        return count + node->meshCount();
    });
}

void Node::destroy(VkDevice device)
{
    if (mesh) {
        mesh->destroy(device);
        delete mesh;
        mesh = nullptr;
    }
    for (auto& child : children) {
        child->destroy(device);
        delete child;
    }
}

gltfModel::gltfModel(std::string filename)
    : filename(filename)
{}

void gltfModel::destroy(VkDevice device)
{
    destroyStagingBuffer(device);

    vertices.destroy(device);
    indices.destroy(device);

    if(descriptorPool != VK_NULL_HANDLE){
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        descriptorPool = VK_NULL_HANDLE;
    }
    if(nodeDescriptorSetLayout != VK_NULL_HANDLE){
        vkDestroyDescriptorSetLayout(device, nodeDescriptorSetLayout,nullptr);
        nodeDescriptorSetLayout = VK_NULL_HANDLE;
    }
    if(materialDescriptorSetLayout != VK_NULL_HANDLE){
        vkDestroyDescriptorSetLayout(device, materialDescriptorSetLayout,nullptr);
        materialDescriptorSetLayout = VK_NULL_HANDLE;
    }

    for (auto& node : nodes){
        node->destroy(device);
        delete node;
    }
    for (auto texture : textures){
        texture.destroy(&device);
    }
    nodes.clear();
    animations.clear();
    textures.clear();
    materials.clear();
};

void gltfModel::destroyStagingBuffer(VkDevice device)
{
    for(auto& texture: textures){
        texture.destroyStagingBuffer(&device);
    }

    vertexStaging.destroy(device);
    indexStaging.destroy(device);
}

const VkBuffer* gltfModel::getVertices() const{
    return &vertices.instance;
}

const VkBuffer* gltfModel::getIndices() const{
    return &indices.instance;
}

void gltfModel::loadNode(VkPhysicalDevice physicalDevice, VkDevice device, Node* parent, const tinygltf::Node &node, uint32_t nodeIndex, const tinygltf::Model &model, std::vector<uint32_t>& indexBuffer, std::vector<Vertex>& vertexBuffer)
{
    Node *newNode = new Node{};
    newNode->index = nodeIndex;
    newNode->parent = parent;
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
            loadNode(physicalDevice, device, newNode, model.nodes[node.children[i]], node.children[i], model, indexBuffer, vertexBuffer);
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
            Mesh::Primitive *newPrimitive = new Mesh::Primitive(indexStart, indexCount, vertexCount, primitive.material > -1 ? materials[primitive.material] : materials.back());
            newPrimitive->bb = BoundingBox(posMin, posMax);
            newMesh->primitives.push_back(newPrimitive);
        }
        newNode->mesh = newMesh;
    }
    if (parent) {
        parent->children.push_back(newNode);
    } else {
        nodes.push_back(newNode);
    }
}

void gltfModel::loadSkins(tinygltf::Model &gltfModel){
    for (const tinygltf::Skin& source: gltfModel.skins) {
        std::shared_ptr<Node::Skin> newSkin(new Node::Skin{});

        for (const auto& node: gltfModel.nodes) {
            if(node.skin == &source - &gltfModel.skins[0]){
                nodeFromIndex(&node - &gltfModel.nodes[0], nodes)->skin = newSkin;
            }
        }

        for (int jointIndex : source.joints) {
            if (Node* node = nodeFromIndex(jointIndex, nodes); node) {
                newSkin->joints.push_back(node);
            }
        }

        if (source.inverseBindMatrices > -1) {
            const tinygltf::Accessor& accessor = gltfModel.accessors[source.inverseBindMatrices];
            const tinygltf::BufferView& bufferView = gltfModel.bufferViews[accessor.bufferView];
            const tinygltf::Buffer& buffer = gltfModel.buffers[bufferView.buffer];

            newSkin->inverseBindMatrices.resize(accessor.count);
            std::memcpy(newSkin->inverseBindMatrices.data(), &buffer.data[accessor.byteOffset + bufferView.byteOffset], accessor.count * sizeof(glm::mat4));
        }
    }
}

void gltfModel::loadTextures(VkPhysicalDevice physicalDevice, VkDevice device, VkCommandBuffer commandBuffer, tinygltf::Model& gltfModel)
{
    for(tinygltf::Texture &tex : gltfModel.textures){
        textureSampler TextureSampler{};
            TextureSampler.minFilter = tex.sampler == -1 ? VK_FILTER_LINEAR : getVkFilterMode(gltfModel.samplers[tex.sampler].minFilter);
            TextureSampler.magFilter = tex.sampler == -1 ? VK_FILTER_LINEAR : getVkFilterMode(gltfModel.samplers[tex.sampler].magFilter);
            TextureSampler.addressModeU = tex.sampler == -1 ? VK_SAMPLER_ADDRESS_MODE_REPEAT : getVkWrapMode(gltfModel.samplers[tex.sampler].wrapS);
            TextureSampler.addressModeV = tex.sampler == -1 ? VK_SAMPLER_ADDRESS_MODE_REPEAT : getVkWrapMode(gltfModel.samplers[tex.sampler].wrapT);
            TextureSampler.addressModeW = tex.sampler == -1 ? VK_SAMPLER_ADDRESS_MODE_REPEAT : TextureSampler.addressModeV;
        textures.emplace_back(texture{});
        textures.back().createTextureImage(physicalDevice,device,commandBuffer,gltfModel.images[tex.source]);
        textures.back().createTextureImageView(&device);
        textures.back().createTextureSampler(&device,TextureSampler);
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

        // Samplers
        for (auto &samp : anim.samplers) {
            Animation::AnimationSampler sampler{};

            if (samp.interpolation == "LINEAR") {
                sampler.interpolation = Animation::AnimationSampler::InterpolationType::LINEAR;
            }
            if (samp.interpolation == "STEP") {
                sampler.interpolation = Animation::AnimationSampler::InterpolationType::STEP;
            }
            if (samp.interpolation == "CUBICSPLINE") {
                sampler.interpolation = Animation::AnimationSampler::InterpolationType::CUBICSPLINE;
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
            Animation::AnimationChannel channel{};

            if (source.target_path == "rotation") {
                channel.path = Animation::AnimationChannel::PathType::ROTATION;
            }
            if (source.target_path == "translation") {
                channel.path = Animation::AnimationChannel::PathType::TRANSLATION;
            }
            if (source.target_path == "scale") {
                channel.path = Animation::AnimationChannel::PathType::SCALE;
            }
            if (source.target_path == "weights") {
                std::cout << "weights not yet supported, skipping channel" << std::endl;
                continue;
            }
            channel.samplerIndex = source.sampler;
            channel.node = nodeFromIndex(source.target_node, nodes);
            if (!channel.node) {
                continue;
            }

            animation.channels.push_back(channel);
        }

        animations.push_back(animation);
    }
}

void gltfModel::loadFromFile(VkPhysicalDevice physicalDevice, VkDevice device, VkCommandBuffer commandBuffer)
{
    tinygltf::Model gltfModel;
    tinygltf::TinyGLTF gltfContext;
    std::string error{}, warning{};

    if (isBinary(filename) ? gltfContext.LoadBinaryFromFile(&gltfModel, &error, &warning, filename.c_str()) : gltfContext.LoadASCIIFromFile(&gltfModel, &error, &warning, filename.c_str()))
    {
        loadTextures(physicalDevice,device,commandBuffer,gltfModel);
        loadMaterials(gltfModel);

        std::vector<uint32_t> indexBuffer;
        std::vector<Vertex> vertexBuffer;

        for (const auto& node: gltfModel.scenes[gltfModel.defaultScene > -1 ? gltfModel.defaultScene : 0].nodes) {
            loadNode(physicalDevice, device, nullptr, gltfModel.nodes[node], node, gltfModel, indexBuffer, vertexBuffer);
        }
        loadSkins(gltfModel);
        if (gltfModel.animations.size() > 0) {
            loadAnimations(gltfModel);
        }

        for (auto& node : nodes) {
            calculateNodeTangent(node, vertexBuffer, indexBuffer);
            node->update();
        }

        size_t vertexBufferSize = vertexBuffer.size() * sizeof(Vertex);
        size_t indexBufferSize = indexBuffer.size() * sizeof(uint32_t);

        Buffer::create(physicalDevice, device,vertexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &vertexStaging.instance, &vertexStaging.memory);
        Buffer::create(physicalDevice, device,indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &indexStaging.instance, &indexStaging.memory);
        Buffer::create(physicalDevice, device,vertexBufferSize,VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &vertices.instance, &vertices.memory);
        Buffer::create(physicalDevice, device,indexBufferSize,VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &indices.instance, &indices.memory);

        vkMapMemory(device, vertexStaging.memory, 0, vertexBufferSize, 0, &vertexStaging.map);
            std::memcpy(vertexStaging.map, vertexBuffer.data(), (size_t) vertexBufferSize);
        vkUnmapMemory(device, vertexStaging.memory);

        vkMapMemory(device, indexStaging.memory, 0, indexBufferSize, 0, &indexStaging.map);
            std::memcpy(indexStaging.map, indexBuffer.data(), (size_t) indexBufferSize);
        vkUnmapMemory(device, indexStaging.memory);

        Buffer::copy(commandBuffer, vertexBufferSize, vertexStaging.instance, vertices.instance);
        Buffer::copy(commandBuffer, indexBufferSize, indexStaging.instance, indices.instance);
    }
}

bool gltfModel::hasAnimation() const {
    return animations.size() > 0;
}

float gltfModel::animationStart(uint32_t index) const {
    return animations[index].start;
}

float gltfModel::animationEnd(uint32_t index) const {
    return animations[index].end;
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
        Animation::AnimationSampler &sampler = animation.samplers[channel.samplerIndex];
        if (sampler.inputs.size() > sampler.outputsVec4.size()) {
            continue;
        }

        for (size_t i = 0; i < sampler.inputs.size() - 1; i++) {
            if ((time >= sampler.inputs[i]) && (time <= sampler.inputs[i + 1])) {
                float u = std::max(0.0f, time - sampler.inputs[i]) / (sampler.inputs[i + 1] - sampler.inputs[i]);
                if (u <= 1.0f) {
                    switch (channel.path) {
                        case Animation::AnimationChannel::PathType::TRANSLATION: {
                            glm::vec4 trans = glm::mix(sampler.outputsVec4[i], sampler.outputsVec4[i + 1], u);
                            channel.node->translation = glm::vec3(trans);
                            break;
                        }
                        case Animation::AnimationChannel::PathType::SCALE: {
                            glm::vec4 trans = glm::mix(sampler.outputsVec4[i], sampler.outputsVec4[i + 1], u);
                            channel.node->scale = glm::vec3(trans);
                            break;
                        }
                        case Animation::AnimationChannel::PathType::ROTATION: {
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
        Animation::AnimationSampler &samplerOld = animationOld.samplers[channel.samplerIndex];
        Animation::AnimationSampler &samplerNew = animationNew.samplers[channel.samplerIndex];
        if (samplerOld.inputs.size() > samplerOld.outputsVec4.size())
            continue;

        for (size_t i = 0; i < samplerOld.inputs.size(); i++) {
            if ((startTime >= samplerOld.inputs[i]) && (time <= samplerOld.inputs[i]+changeAnimationTime)) {
                float u = std::max(0.0f, time - startTime) / changeAnimationTime;
                if (u <= 1.0f) {
                    switch (channel.path) {
                        case Animation::AnimationChannel::PathType::TRANSLATION: {
                            glm::vec4 trans = glm::mix(samplerOld.outputsVec4[i], samplerNew.outputsVec4[0], u);
                            channel.node->translation = glm::vec3(trans);
                            break;
                        }
                        case Animation::AnimationChannel::PathType::SCALE: {
                            glm::vec4 trans = glm::mix(samplerOld.outputsVec4[i], samplerNew.outputsVec4[0], u);
                            channel.node->scale = glm::vec3(trans);
                            break;
                        }
                        case Animation::AnimationChannel::PathType::ROTATION: {
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

void gltfModel::createDescriptorPool(VkDevice device)
{
    uint32_t imageSamplerCount = std::accumulate(materials.begin(), materials.end(), 0, [](const uint32_t& count, const auto& material){
        static_cast<void>(material);
        return count + 5;
    });
    uint32_t meshCount = std::accumulate(nodes.begin(), nodes.end(), 0, [](const uint32_t& count, Node* node){
        return count + node->meshCount();
    });

    std::vector<VkDescriptorPoolSize> poolSize;
    poolSize.push_back({VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, meshCount});
    poolSize.push_back({VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, imageSamplerCount});

    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSize.size());
        poolInfo.pPoolSizes = poolSize.data();
        poolInfo.maxSets = std::max(meshCount, imageSamplerCount);
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);
}

void gltfModel::createDescriptorSet(VkDevice device, texture* emptyTexture)
{
    gltfModel::createMaterialDescriptorSetLayout(device, &materialDescriptorSetLayout);
    gltfModel::createNodeDescriptorSetLayout(device, &nodeDescriptorSetLayout);

    for (auto& node : nodes){
        createNodeDescriptorSet(device, node , descriptorPool, nodeDescriptorSetLayout);
    }

    for (auto &material : materials){
        createMaterialDescriptorSet(device, &material, emptyTexture, descriptorPool, materialDescriptorSetLayout);
    }
}

void renderNode(Node *node, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t& primitiveCount, uint32_t pushConstantSize, uint32_t pushConstantOffset, void* pushConstant)
{
    if (node->mesh)
    {
        for (Mesh::Primitive* primitive : node->mesh->primitives)
        {
            std::vector<VkDescriptorSet> nodeDescriptorSets(descriptorSetsCount);
            std::copy(descriptorSets, descriptorSets + descriptorSetsCount, nodeDescriptorSets.data());
            nodeDescriptorSets.push_back(node->mesh->uniformBuffer.descriptorSet);
            nodeDescriptorSets.push_back(primitive->material.descriptorSet);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, descriptorSetsCount+2, nodeDescriptorSets.data(), 0, NULL);

            MaterialBlock material{};
                material.primitive = primitiveCount++;
                material.emissiveFactor = primitive->material.emissiveFactor;
                material.colorTextureSet = primitive->material.baseColorTexture != nullptr ? primitive->material.texCoordSets.baseColor : -1;
                material.normalTextureSet = primitive->material.normalTexture != nullptr ? primitive->material.texCoordSets.normal : -1;
                material.occlusionTextureSet = primitive->material.occlusionTexture != nullptr ? primitive->material.texCoordSets.occlusion : -1;
                material.emissiveTextureSet = primitive->material.emissiveTexture != nullptr ? primitive->material.texCoordSets.emissive : -1;
                material.alphaMask = static_cast<float>(primitive->material.alphaMode == Material::ALPHAMODE_MASK);
                material.alphaMaskCutoff = primitive->material.alphaCutoff;
            if (primitive->material.pbrWorkflows.metallicRoughness) {
                material.workflow = static_cast<float>(PBR_WORKFLOW_METALLIC_ROUGHNESS);
                material.baseColorFactor = primitive->material.baseColorFactor;
                material.metallicFactor = primitive->material.metallicFactor;
                material.roughnessFactor = primitive->material.roughnessFactor;
                material.PhysicalDescriptorTextureSet = primitive->material.metallicRoughnessTexture != nullptr ? primitive->material.texCoordSets.metallicRoughness : -1;
                material.colorTextureSet = primitive->material.baseColorTexture != nullptr ? primitive->material.texCoordSets.baseColor : -1;
            }
            if (primitive->material.pbrWorkflows.specularGlossiness) {
                material.workflow = static_cast<float>(PBR_WORKFLOW_SPECULAR_GLOSINESS);
                material.PhysicalDescriptorTextureSet = primitive->material.extension.specularGlossinessTexture != nullptr ? primitive->material.texCoordSets.specularGlossiness : -1;
                material.colorTextureSet = primitive->material.extension.diffuseTexture != nullptr ? primitive->material.texCoordSets.baseColor : -1;
                material.diffuseFactor = primitive->material.extension.diffuseFactor;
                material.specularFactor = glm::vec4(primitive->material.extension.specularFactor, 1.0f);
            }
            std::memcpy(reinterpret_cast<char*>(pushConstant) + pushConstantOffset, &material, sizeof(MaterialBlock));

            vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_ALL, 0, pushConstantSize, pushConstant);

            if (primitive->hasIndices){
                vkCmdDrawIndexed(commandBuffer, primitive->indexCount, 1, primitive->firstIndex, 0, 0);
            }else{
                vkCmdDraw(commandBuffer, primitive->vertexCount, 1, 0, 0);
            }
        }
    }
    for (auto child : node->children){
        renderNode(child, commandBuffer, pipelineLayout, descriptorSetsCount, descriptorSets, primitiveCount, pushConstantSize, pushConstantOffset, pushConstant);
    }
}

void gltfModel::render(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t &primitiveCount, uint32_t pushConstantSize, uint32_t pushConstantOffset, void* pushConstant){
    for (auto node: nodes){
        renderNode(node, commandBuffer, pipelineLayout, descriptorSetsCount, descriptorSets, primitiveCount, pushConstantSize, pushConstantOffset, pushConstant);
    }
}

