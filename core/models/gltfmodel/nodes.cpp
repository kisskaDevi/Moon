#include "gltfmodel.h"
#include "operations.h"

#include <numeric>

namespace {
    matrix<float,4,4> localMatrix(Node* node){
        return translate(node->translation) * rotate(node->rotation) * scale(node->scale) * node->matrix;
    }

    matrix<float,4,4> getMatrix(Node* node) {
        return (node->parent ? getMatrix(node->parent) : matrix<float,4,4>(1.0f)) * localMatrix(node);
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
    vector<float,4> loadJoint(const void* bufferJoints, int jointByteStride, size_t vertex){
        const type *buf = static_cast<const type*>(bufferJoints);
        return vector<float,4>(
            buf[vertex * jointByteStride + 0],
            buf[vertex * jointByteStride + 1],
            buf[vertex * jointByteStride + 2],
            buf[vertex * jointByteStride + 3]
        );
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
            buffer.second = accessor.ByteStride(view) ? (accessor.ByteStride(view) / static_cast<int>(size)) : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE);
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
        mesh->destroy(device); delete mesh; mesh = nullptr;
    }
    for (auto& child : children) {
        child->destroy(device);
        delete child;
    }
}

Mesh::Primitive::Primitive(uint32_t firstIndex, uint32_t indexCount, uint32_t vertexCount, Material* material, BoundingBox bb)
    : firstIndex(firstIndex), indexCount(indexCount), vertexCount(vertexCount), material(material), bb(bb)
{}

Mesh::Mesh(VkPhysicalDevice physicalDevice, VkDevice device, matrix<float,4,4> matrix)
{
    this->uniformBlock.matrix = matrix;
    moon::utils::buffer::create( physicalDevice,
                    device,
                    sizeof(uniformBlock),
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    &uniformBuffer.instance,
                    &uniformBuffer.memory);
    CHECK(vkMapMemory(device, uniformBuffer.memory, 0, sizeof(uniformBlock), 0, &uniformBuffer.map));

    moon::utils::Memory::instance().nameMemory(uniformBuffer.memory, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", Mesh::Mesh, uniformBuffer");
};

void Mesh::destroy(VkDevice device){
    uniformBuffer.destroy(device);
    for (Primitive* p : primitives){
        delete p;
    }
}

void Node::update() {
    if (size_t numJoints = skin ? std::min((uint32_t)skin->joints.size(), MAX_NUM_JOINTS) : 0; mesh){
        mesh->uniformBlock.matrix = transpose(getMatrix(this));

        for (size_t i = 0; i < numJoints; i++) {
            mesh->uniformBlock.jointMatrix[i] = transpose(inverse(getMatrix(this)) * getMatrix(skin->joints[i]) * skin->inverseBindMatrices[i]);
        }
        mesh->uniformBlock.jointcount = static_cast<float>(numJoints);
        std::memcpy(mesh->uniformBuffer.map, &mesh->uniformBlock, sizeof(mesh->uniformBlock));
    }
    for (auto& child : children){
        child->update();
    }
}

uint32_t Node::meshCount() const {
    return std::accumulate(children.begin(), children.end(), mesh ? 1 : 0, [](const uint32_t& count, Node* node){
        return count + node->meshCount();
    });
}

void gltfModel::loadNode(instance* instance, VkPhysicalDevice physicalDevice, VkDevice device, Node* parent, uint32_t nodeIndex, const tinygltf::Model &model, uint32_t& indexStart)
{
    Node* newNode = new Node{};
    newNode->index = nodeIndex;
    newNode->parent = parent;
    newNode->matrix = matrix<float,4,4>(1.0f);

    if (model.nodes[nodeIndex].translation.size() == 3) {
        newNode->translation = vector<float,3>(
            static_cast<float>(model.nodes[nodeIndex].translation[0]),
            static_cast<float>(model.nodes[nodeIndex].translation[1]),
            static_cast<float>(model.nodes[nodeIndex].translation[2])
        );
    }
    if (model.nodes[nodeIndex].rotation.size() == 4) {
        newNode->rotation = quaternion<float>(
            static_cast<float>(model.nodes[nodeIndex].rotation[3]),
            static_cast<float>(model.nodes[nodeIndex].rotation[0]),
            static_cast<float>(model.nodes[nodeIndex].rotation[1]),
            static_cast<float>(model.nodes[nodeIndex].rotation[2])
        );
    }
    if (model.nodes[nodeIndex].scale.size() == 3) {
        newNode->scale = vector<float,3>(
            static_cast<float>(model.nodes[nodeIndex].scale[0]),
            static_cast<float>(model.nodes[nodeIndex].scale[1]),
            static_cast<float>(model.nodes[nodeIndex].scale[2])
        );
    }
    if (model.nodes[nodeIndex].matrix.size() == 16) {
        const double* m = model.nodes[nodeIndex].matrix.data();
        newNode->matrix = matrix<float,4,4>(0.0f);
        for(uint32_t i = 0; i < 4; i++){
            for(uint32_t j = 0; j < 4; j++)
            newNode->matrix[i][j] = static_cast<float>(m[4*i + j]);
        }
    }

    for (const auto& children: model.nodes[nodeIndex].children) {
        loadNode(instance, physicalDevice, device, newNode, children, model, indexStart);
    }

    if (model.nodes[nodeIndex].mesh > -1) {
        newNode->mesh = new Mesh(physicalDevice,device,newNode->matrix);
        for (const tinygltf::Primitive &primitive: model.meshes[model.nodes[nodeIndex].mesh].primitives)
        {
            const tinygltf::Accessor& posAccessor = model.accessors[primitive.attributes.find("POSITION")->second];

            uint32_t indexCount = primitive.indices > -1 ? static_cast<uint32_t>(model.accessors[primitive.indices > -1 ? primitive.indices : 0].count) : 0;
            uint32_t vertexCount = static_cast<uint32_t>(posAccessor.count);

            newNode->mesh->primitives.push_back(
                new Mesh::Primitive(indexStart, indexCount, vertexCount, primitive.material > -1 ? &materials[primitive.material] : &materials.back(),
                    BoundingBox(
                        vector<float,3>(static_cast<float>(posAccessor.minValues[0]), static_cast<float>(posAccessor.minValues[1]), static_cast<float>(posAccessor.minValues[2])),
                        vector<float,3>(static_cast<float>(posAccessor.maxValues[0]), static_cast<float>(posAccessor.maxValues[1]), static_cast<float>(posAccessor.maxValues[2]))
                    )
                )
            );

            if (primitive.indices > -1){
                indexStart += static_cast<uint32_t>(model.accessors[primitive.indices > -1 ? primitive.indices : 0].count);
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
    for (const auto& children: node.children) {
        loadVertexBuffer(model.nodes[children], model, indexBuffer, vertexBuffer);
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
                vert.pos = pos.first ? vector<float,3>(pos.first[index * pos.second + 0], pos.first[index * pos.second + 1], pos.first[index * pos.second + 2]) : vector<float,3>(0.0f);
                vert.normal = normalize(vector<float,3>(normals.first ? vector<float,3>(normals.first[index * normals.second], normals.first[index * normals.second + 1], normals.first[index * normals.second + 2]) : vector<float,3>(0.0f)));
                vert.uv0 = texCoordSet0.first ? vector<float,2>(texCoordSet0.first[index * texCoordSet0.second], texCoordSet0.first[index * texCoordSet0.second + 1]) : vector<float,2>(0.0f);
                vert.uv1 = texCoordSet1.first ? vector<float,2>(texCoordSet1.first[index * texCoordSet1.second], texCoordSet1.first[index * texCoordSet1.second + 1]) : vector<float,2>(0.0f);
                vert.joint0 = vector<float,4>(0.0f, 0.0f, 0.0f, 0.0f);
                vert.weight0 = vector<float,4>(1.0f, 0.0f, 0.0f, 0.0f);

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

                    vert.weight0 = vector<float,4>(
                        weights.first[index * weights.second + 0],
                        weights.first[index * weights.second + 1],
                        weights.first[index * weights.second + 2],
                        weights.first[index * weights.second + 3]
                    );
                    if (dot(vert.weight0,vert.weight0) == 0.0f) {
                        vert.weight0 = vector<float,4>(1.0f, 0.0f, 0.0f, 0.0f);
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
