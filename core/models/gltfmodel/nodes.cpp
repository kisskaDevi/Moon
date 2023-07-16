#include "gltfmodel.h"
#include "operations.h"

#include <iostream>
#include <numeric>

namespace {
    glm::mat4 localMatrix(Node* node){
        return glm::translate(glm::mat4(1.0f), node->translation) * glm::mat4(node->rotation) * glm::scale(glm::mat4(1.0f), node->scale) * node->matrix;
    }

    glm::mat4 getMatrix(Node* node) {
        return (node->parent ? getMatrix(node->parent) : glm::mat4(1.0f)) * localMatrix(node);
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

    template <typename type>
    glm::vec4 loadJoint(const void* bufferJoints, int jointByteStride, size_t vertex){
        const type *buf = static_cast<const type*>(bufferJoints);
        return glm::vec4(glm::make_vec4(&buf[vertex * jointByteStride]));
    }

    template <typename type>
    void pushBackIndex(const void *dataPtr, size_t count, uint32_t vertexStart, std::vector<uint32_t>& indexBuffer){
        const type *buf = static_cast<const type*>(dataPtr);
        for (size_t index = 0; index < count; index++) {
            indexBuffer.push_back(buf[index] + vertexStart);
        }
    }

    template <typename type>
    std::pair<const type*, int> loadBuffer(const tinygltf::Primitive& primitive, const tinygltf::Model& model, std::string attribute, uint64_t size, int TINYGLTF_TYPE){
        std::pair<const type*, int> buffer;
        if (primitive.attributes.find(attribute) != primitive.attributes.end()) {
            const tinygltf::Accessor &accessor = model.accessors[primitive.attributes.find(attribute)->second];
            const tinygltf::BufferView &view = model.bufferViews[accessor.bufferView];
            buffer.first = reinterpret_cast<const float *>(&(model.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
            buffer.second = accessor.ByteStride(view) ? (accessor.ByteStride(view) / size) : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE);
        }
        return buffer;
    }
}

Node* gltfModel::nodeFromIndex(uint32_t index, const std::vector<Node*>& nodes)
{
    Node* nodeFound = nullptr;
    for (const auto &node : nodes) {
        nodeFound = findNodeFromIndex(node, index);
        if (nodeFound) break;
    }
    return nodeFound;
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

Mesh::Primitive::Primitive(uint32_t firstIndex, uint32_t indexCount, uint32_t vertexCount, Material* material)
    : firstIndex(firstIndex), indexCount(indexCount), vertexCount(vertexCount), material(material)
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

void gltfModel::loadNode(instance* instance, VkPhysicalDevice physicalDevice, VkDevice device, Node* parent, uint32_t nodeIndex, const tinygltf::Model &model, uint32_t& indexStart)
{
    Node* newNode = new Node{};
    newNode->index = nodeIndex;
    newNode->parent = parent;
    newNode->matrix = glm::mat4(1.0f);

    if (model.nodes[nodeIndex].translation.size() == 3) {
        newNode->translation = glm::make_vec3(model.nodes[nodeIndex].translation.data());
    }
    if (model.nodes[nodeIndex].rotation.size() == 4) {
        newNode->rotation = glm::mat4(static_cast<glm::quat>(glm::make_quat(model.nodes[nodeIndex].rotation.data())));
    }
    if (model.nodes[nodeIndex].scale.size() == 3) {
        newNode->scale = glm::make_vec3(model.nodes[nodeIndex].scale.data());
    }
    if (model.nodes[nodeIndex].matrix.size() == 16) {
        newNode->matrix = glm::make_mat4x4(model.nodes[nodeIndex].matrix.data());
    }

    for (size_t i = 0; i < model.nodes[nodeIndex].children.size(); i++) {
        loadNode(instance, physicalDevice, device, newNode, model.nodes[nodeIndex].children[i], model, indexStart);
    }

    if (model.nodes[nodeIndex].mesh > -1) {
        const tinygltf::Mesh mesh = model.meshes[model.nodes[nodeIndex].mesh];
        Mesh* newMesh = new Mesh(physicalDevice,device,newNode->matrix);
        newNode->mesh = newMesh;
        for (const tinygltf::Primitive &primitive: mesh.primitives)
        {
            const tinygltf::Accessor& posAccessor = model.accessors[primitive.attributes.find("POSITION")->second];

            uint32_t indexCount = primitive.indices > -1 ? model.accessors[primitive.indices > -1 ? primitive.indices : 0].count : 0;
            uint32_t vertexCount = static_cast<uint32_t>(posAccessor.count);

            Mesh::Primitive* newPrimitive = new Mesh::Primitive(indexStart, indexCount, vertexCount, primitive.material > -1 ? &materials[primitive.material] : &materials.back());
            newPrimitive->bb = BoundingBox( glm::vec3(posAccessor.minValues[0], posAccessor.minValues[1], posAccessor.minValues[2]),
                                            glm::vec3(posAccessor.maxValues[0], posAccessor.maxValues[1], posAccessor.maxValues[2]));
            newMesh->primitives.push_back(newPrimitive);

            if (primitive.indices > -1){
                indexStart += model.accessors[primitive.indices > -1 ? primitive.indices : 0].count;
            }
        }
    }
    if (parent) {
        parent->children.push_back(newNode);
    } else {
        instance->nodes.push_back(newNode);
    }
}

void gltfModel::loadVertexBuffer(const tinygltf::Node& node, const tinygltf::Model& model, std::vector<uint32_t>& indexBuffer, std::vector<Vertex>& vertexBuffer)
{
    for (size_t i = 0; i < node.children.size(); i++) {
        loadVertexBuffer(model.nodes[node.children[i]], model, indexBuffer, vertexBuffer);
    }

    if (node.mesh > -1) {
        const tinygltf::Mesh mesh = model.meshes[node.mesh];
        for (const tinygltf::Primitive &primitive: mesh.primitives)
        {
            const tinygltf::Accessor& posAccessor = model.accessors[primitive.attributes.find("POSITION")->second];

            uint32_t vertexStart = static_cast<uint32_t>(vertexBuffer.size());
            uint32_t vertexCount = static_cast<uint32_t>(posAccessor.count);

            std::pair<const float*, int> pos = loadBuffer<float>(primitive, model, "POSITION", sizeof(float), TINYGLTF_TYPE_VEC3);
            std::pair<const float*, int> normals = loadBuffer<float>(primitive, model, "NORMAL", sizeof(float), TINYGLTF_TYPE_VEC3);
            std::pair<const float*, int> texCoordSet0 = loadBuffer<float>(primitive, model, "TEXCOORD_0", sizeof(float), TINYGLTF_TYPE_VEC2);
            std::pair<const float*, int> texCoordSet1 = loadBuffer<float>(primitive, model, "TEXCOORD_1", sizeof(float), TINYGLTF_TYPE_VEC2);
            std::pair<const void*, int> joints = { nullptr, 0 };
            if (auto jointsAttr = primitive.attributes.find("JOINTS_0"); jointsAttr != primitive.attributes.end()) {
                joints = loadBuffer<void>(primitive, model, "JOINTS_0", tinygltf::GetComponentSizeInBytes(model.accessors[jointsAttr->second].componentType), TINYGLTF_TYPE_VEC4);
            }
            std::pair<const float*, int> weights = loadBuffer<float>(primitive, model, "WEIGHTS_0", sizeof(float), TINYGLTF_TYPE_VEC4);

            for (uint32_t index = 0; index < vertexCount; index++) {
                Vertex vert{};
                vert.pos = glm::vec4(glm::make_vec3(&pos.first[index * pos.second]), 1.0f);
                vert.normal = glm::normalize(glm::vec3(normals.first ? glm::make_vec3(&normals.first[index * normals.second]) : glm::vec3(0.0f)));
                vert.uv0 = texCoordSet0.first ? glm::make_vec2(&texCoordSet0.first[index * texCoordSet0.second]) : glm::vec2(0.0f);
                vert.uv1 = texCoordSet1.first ? glm::make_vec2(&texCoordSet1.first[index * texCoordSet1.second]) : glm::vec2(0.0f);
                vert.joint0 = glm::vec4(0.0f);
                vert.weight0 = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);

                if (joints.first && weights.first)
                {
                    switch (model.accessors[primitive.attributes.find("JOINTS_0")->second].componentType) {
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                            vert.joint0 = loadJoint<uint16_t>(joints.first, joints.second, index);
                            break;
                        }
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
                            vert.joint0 = loadJoint<uint8_t>(joints.first, joints.second, index);
                            break;
                        }
                    }

                    vert.weight0 = glm::make_vec4(&weights.first[index * weights.second]);
                    if (glm::length(vert.weight0) == 0.0f) {
                        vert.weight0 = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
                    }
                }
                vertexBuffer.push_back(vert);
            }

            if (primitive.indices > -1)
            {
                const tinygltf::Accessor &accessor = model.accessors[primitive.indices > -1 ? primitive.indices : 0];
                const tinygltf::BufferView &bufferView = model.bufferViews[accessor.bufferView];
                const void *dataPtr = &(model.buffers[bufferView.buffer].data[accessor.byteOffset + bufferView.byteOffset]);

                switch (accessor.componentType) {
                    case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT: {
                        pushBackIndex<uint32_t>(dataPtr, accessor.count, vertexStart, indexBuffer);
                        break;
                    }
                    case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT: {
                        pushBackIndex<uint16_t>(dataPtr, accessor.count, vertexStart, indexBuffer);
                        break;
                    }
                    case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE: {
                        pushBackIndex<uint8_t>(dataPtr, accessor.count, vertexStart, indexBuffer);
                        break;
                    }
                }
            }
        }
    }
}
