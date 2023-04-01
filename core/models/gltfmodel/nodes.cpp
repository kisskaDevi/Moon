#include "../gltfmodel.h"
#include "../../utils/operations.h"

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
        for (size_t j = 0; j < mesh.primitives.size(); j++)
        {
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
            Mesh::Primitive* newPrimitive = new Mesh::Primitive(indexStart, indexCount, vertexCount, primitive.material > -1 ? &materials[primitive.material] : &materials.back());
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
