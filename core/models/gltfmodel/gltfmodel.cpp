#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define STBI_MSC_SECURE_CRT

#include "gltfmodel.h"
#include "operations.h"

#include <iostream>
#include <fstream>
#include <cstring>

namespace {

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

    void calculateNodeTangent(std::vector<model::Vertex>& vertexBuffer, std::vector<uint32_t>& indexBuffer){
        for(uint32_t i = 0; i < indexBuffer.size(); i += 3){
            glm::vec3 dv1   = vertexBuffer[indexBuffer[i+1]].pos - vertexBuffer[indexBuffer[i+0]].pos;
            glm::vec3 dv2   = vertexBuffer[indexBuffer[i+2]].pos - vertexBuffer[indexBuffer[i+0]].pos;

            glm::vec2 duv1  = vertexBuffer[indexBuffer[i+1]].uv0 - vertexBuffer[indexBuffer[i+0]].uv0;
            glm::vec2 duv2  = vertexBuffer[indexBuffer[i+2]].uv0 - vertexBuffer[indexBuffer[i+0]].uv0;

            float det = 1.0f/(duv1.x*duv2.y - duv1.y*duv2.x);

            glm::vec3 tangent = glm::normalize( det*(duv2.y*dv1-duv1.y*dv2));
            glm::vec3 bitangent = glm::normalize( det*(duv1.x*dv2-duv2.x*dv1));

            if(dot(glm::cross(tangent,bitangent),vertexBuffer[indexBuffer[i+0]].normal)<0.0f){
                tangent = -tangent;
            }

            for(uint32_t index = 0; index < 3; index++){
                vertexBuffer[indexBuffer[i+index]].tangent      = glm::normalize(tangent - vertexBuffer[indexBuffer[i+index]].normal * glm::dot(vertexBuffer[indexBuffer[i+index]].normal, tangent));
                vertexBuffer[indexBuffer[i+index]].bitangent    = glm::normalize(glm::cross(vertexBuffer[indexBuffer[i+index]].normal, vertexBuffer[indexBuffer[i+index]].tangent));
            }
        }
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

            VkDescriptorBufferInfo bufferInfo{ node->mesh->uniformBuffer.instance, 0, sizeof(Mesh::uniformBlock)};

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

        auto getDescriptorImageInfo = [&emptyTexture](texture* tex){
            VkDescriptorImageInfo descriptorImageInfo{};
            descriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            descriptorImageInfo.imageView   = tex ? *tex->getTextureImageView() : *emptyTexture->getTextureImageView();
            descriptorImageInfo.sampler     = tex ? *tex->getTextureSampler()   : *emptyTexture->getTextureSampler();
            return descriptorImageInfo;
        };

        VkDescriptorImageInfo baseColorTextureInfo{};
        if (material->pbrWorkflows.metallicRoughness){
            baseColorTextureInfo = getDescriptorImageInfo(material->baseColorTexture);
        }
        if(material->pbrWorkflows.specularGlossiness){
            baseColorTextureInfo = getDescriptorImageInfo(material->extension.diffuseTexture);
        }

        VkDescriptorImageInfo metallicRoughnessTextureInfo{};
        if (material->pbrWorkflows.metallicRoughness){
            metallicRoughnessTextureInfo = getDescriptorImageInfo(material->metallicRoughnessTexture);
        }
        if (material->pbrWorkflows.specularGlossiness){
            metallicRoughnessTextureInfo = getDescriptorImageInfo(material->extension.specularGlossinessTexture);
        }

        std::vector<VkDescriptorImageInfo> descriptorImageInfos = {
            baseColorTextureInfo,
            metallicRoughnessTextureInfo,
            getDescriptorImageInfo(material->normalTexture),
            getDescriptorImageInfo(material->occlusionTexture),
            getDescriptorImageInfo(material->emissiveTexture)
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

    void createBuffer(VkPhysicalDevice physicalDevice, VkDevice device, VkCommandBuffer commandBuffer, size_t bufferSize, void* data, VkBufferUsageFlagBits usage, buffer& staging, buffer& deviceLocal){
        Buffer::create(physicalDevice, device, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &staging.instance, &staging.memory);
        Buffer::create(physicalDevice, device, bufferSize, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &deviceLocal.instance, &deviceLocal.memory);

        vkMapMemory(device, staging.memory, 0, bufferSize, 0, &staging.map);
            std::memcpy(staging.map, data, bufferSize);
        vkUnmapMemory(device, staging.memory);
        staging.map = nullptr;

        Buffer::copy(commandBuffer, bufferSize, staging.instance, deviceLocal.instance);
    };
}

BoundingBox::BoundingBox(glm::vec3 min, glm::vec3 max)
    : min(min), max(max), valid(true)
{};

BoundingBox BoundingBox::getAABB(glm::mat4 m) const {
    return BoundingBox(
        glm::min(glm::vec3(m[0]) * this->min.x, glm::vec3(m[0]) * this->max.x) +
        glm::min(glm::vec3(m[1]) * this->min.y, glm::vec3(m[1]) * this->max.y) +
        glm::min(glm::vec3(m[2]) * this->min.z, glm::vec3(m[2]) * this->max.z),
        glm::max(glm::vec3(m[0]) * this->min.x, glm::vec3(m[0]) * this->max.x) +
        glm::max(glm::vec3(m[1]) * this->min.y, glm::vec3(m[1]) * this->max.y) +
        glm::max(glm::vec3(m[2]) * this->min.z, glm::vec3(m[2]) * this->max.z)
    );
}

gltfModel::gltfModel(std::string filename, uint32_t instanceCount)
    : filename(filename)
{
    instances.resize(instanceCount);
}

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

    for(auto& instance : instances){
        for (auto& node : instance.nodes){
            node->destroy(device);
            delete node;
        }
        for (auto& skin : instance.skins){
            delete skin;
        }
        instance.nodes.clear();
        instance.skins.clear();
        instance.animations.clear();
    }
    for (auto texture : textures){
        texture.destroy(device);
    }
    textures.clear();
    materials.clear();
};

void gltfModel::destroyStagingBuffer(VkDevice device)
{
    for(auto& texture: textures){
        texture.destroyStagingBuffer(device);
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

void gltfModel::loadSkins(tinygltf::Model &gltfModel){
    for(auto& instance : instances){
        for (const tinygltf::Skin& source: gltfModel.skins) {
            Skin* newSkin = new Skin{};

            for (int jointIndex : source.joints) {
                if (Node* node = nodeFromIndex(jointIndex, instance.nodes); node) {
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

            for (const auto& node: gltfModel.nodes) {
                if(node.skin == &source - &gltfModel.skins[0]){
                    nodeFromIndex(&node - &gltfModel.nodes[0], instance.nodes)->skin = newSkin;
                }
            }

            instance.skins.push_back(newSkin);
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
        textures.back().createTextureImageView(device);
        textures.back().createTextureSampler(device,TextureSampler);
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

void gltfModel::loadFromFile(VkPhysicalDevice physicalDevice, VkDevice device, VkCommandBuffer commandBuffer)
{
    tinygltf::Model gltfModel;
    tinygltf::TinyGLTF gltfContext;

    if (std::string error{}, warning{}; isBinary(filename) ? gltfContext.LoadBinaryFromFile(&gltfModel, &error, &warning, filename.c_str()) : gltfContext.LoadASCIIFromFile(&gltfModel, &error, &warning, filename.c_str()))
    {
        loadTextures(physicalDevice,device,commandBuffer,gltfModel);
        loadMaterials(gltfModel);

        for(auto& instance: instances){
            uint32_t indexStart = 0;
            for (const auto& nodeIndex: gltfModel.scenes[gltfModel.defaultScene > -1 ? gltfModel.defaultScene : 0].nodes) {
                loadNode(&instance, physicalDevice, device, nullptr, nodeIndex, gltfModel, indexStart);
            }
        }

        std::vector<uint32_t> indexBuffer;
        std::vector<Vertex> vertexBuffer;
        for (const auto& nodeIndex: gltfModel.scenes[gltfModel.defaultScene > -1 ? gltfModel.defaultScene : 0].nodes) {
            loadVertexBuffer(gltfModel.nodes[nodeIndex], gltfModel, indexBuffer, vertexBuffer);
        }
        calculateNodeTangent(vertexBuffer, indexBuffer);

        loadSkins(gltfModel);
        if (gltfModel.animations.size() > 0) {
            loadAnimations(gltfModel);
        }

        for(auto& instance : instances){
            for (auto& node : instance.nodes) {
                node->update();
            }
        }

        createBuffer(physicalDevice, device, commandBuffer, vertexBuffer.size() * sizeof(Vertex), vertexBuffer.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexStaging, vertices);
        createBuffer(physicalDevice, device, commandBuffer, indexBuffer.size() * sizeof(uint32_t), indexBuffer.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexStaging, indices);
    }
}

void gltfModel::createDescriptorPool(VkDevice device)
{
    uint32_t imageSamplerCount = std::accumulate(materials.begin(), materials.end(), 0, [](const uint32_t& count, const auto& material){
        static_cast<void>(material);
        return count + 5;
    });
    uint32_t meshCount = std::accumulate(instances.begin(), instances.end(), 0, [](const uint32_t& count, const auto& instance){
        return count + std::accumulate(instance.nodes.begin(), instance.nodes.end(), 0, [](const uint32_t& count, Node* node){
            return count + node->meshCount();
        });
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

    for(auto& instance : instances){
        for (auto& node : instance.nodes){
            createNodeDescriptorSet(device, node , descriptorPool, nodeDescriptorSetLayout);
        }
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
            nodeDescriptorSets.push_back(primitive->material->descriptorSet);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, descriptorSetsCount+2, nodeDescriptorSets.data(), 0, NULL);

            MaterialBlock material{};
                material.primitive = primitiveCount++;
                material.emissiveFactor = primitive->material->emissiveFactor;
                material.colorTextureSet = primitive->material->baseColorTexture != nullptr ? primitive->material->texCoordSets.baseColor : -1;
                material.normalTextureSet = primitive->material->normalTexture != nullptr ? primitive->material->texCoordSets.normal : -1;
                material.occlusionTextureSet = primitive->material->occlusionTexture != nullptr ? primitive->material->texCoordSets.occlusion : -1;
                material.emissiveTextureSet = primitive->material->emissiveTexture != nullptr ? primitive->material->texCoordSets.emissive : -1;
                material.alphaMask = static_cast<float>(primitive->material->alphaMode == Material::ALPHAMODE_MASK);
                material.alphaMaskCutoff = primitive->material->alphaCutoff;
            if (primitive->material->pbrWorkflows.metallicRoughness) {
                material.workflow = static_cast<float>(PBR_WORKFLOW_METALLIC_ROUGHNESS);
                material.baseColorFactor = primitive->material->baseColorFactor;
                material.metallicFactor = primitive->material->metallicFactor;
                material.roughnessFactor = primitive->material->roughnessFactor;
                material.PhysicalDescriptorTextureSet = primitive->material->metallicRoughnessTexture != nullptr ? primitive->material->texCoordSets.metallicRoughness : -1;
                material.colorTextureSet = primitive->material->baseColorTexture != nullptr ? primitive->material->texCoordSets.baseColor : -1;
            }
            if (primitive->material->pbrWorkflows.specularGlossiness) {
                material.workflow = static_cast<float>(PBR_WORKFLOW_SPECULAR_GLOSINESS);
                material.PhysicalDescriptorTextureSet = primitive->material->extension.specularGlossinessTexture != nullptr ? primitive->material->texCoordSets.specularGlossiness : -1;
                material.colorTextureSet = primitive->material->extension.diffuseTexture != nullptr ? primitive->material->texCoordSets.baseColor : -1;
                material.diffuseFactor = primitive->material->extension.diffuseFactor;
                material.specularFactor = glm::vec4(primitive->material->extension.specularFactor, 1.0f);
            }
            std::memcpy(reinterpret_cast<char*>(pushConstant) + pushConstantOffset, &material, sizeof(MaterialBlock));

            vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_ALL, 0, pushConstantSize, pushConstant);

            if (primitive->indexCount > 0){
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

void gltfModel::render(uint32_t frameIndex, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t &primitiveCount, uint32_t pushConstantSize, uint32_t pushConstantOffset, void* pushConstant){
    for (auto node: instances[frameIndex].nodes){
        renderNode(node, commandBuffer, pipelineLayout, descriptorSetsCount, descriptorSets, primitiveCount, pushConstantSize, pushConstantOffset, pushConstant);
    }
}

