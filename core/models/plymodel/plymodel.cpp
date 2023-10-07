#include "plymodel.h"

#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"
#include "operations.h"

#include <memory>
#include <fstream>
#include <iostream>
#include <vector>

plyModel::plyModel(
        std::filesystem::path filename,
        vector<float, 4> baseColorFactor,
        vector<float, 4> diffuseFactor,
        vector<float, 4> specularFactor,
        float metallicFactor,
        float roughnessFactor,
        float workflow) : filename(filename)
{
    materialBlock.baseColorFactor = baseColorFactor;
    materialBlock.diffuseFactor = diffuseFactor;
    materialBlock.specularFactor = specularFactor;
    materialBlock.metallicFactor = metallicFactor;
    materialBlock.roughnessFactor = roughnessFactor;
    materialBlock.workflow = workflow;
}

plyModel::~plyModel() {}

void plyModel::destroy(VkDevice device) {
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

    uniformBuffer.destroy(device);
}
void plyModel::destroyStagingBuffer(VkDevice device) {
    vertexStaging.destroy(device);
    indexStaging.destroy(device);
}

const VkBuffer* plyModel::getVertices() const {
    return &vertices.instance;
}
const VkBuffer* plyModel::getIndices() const {
    return &indices.instance;
}
const vector<float,3> plyModel::getMaxSize() const {
    return maxSize;
}

void plyModel::loadFromFile(VkPhysicalDevice physicalDevice, VkDevice device, VkCommandBuffer commandBuffer) {
    tinyply::PlyFile file;
    std::ifstream file_stream(filename, std::ios::binary);
    file.parse_header(file_stream);

    std::shared_ptr<tinyply::PlyData> vertices, normals, texcoords, faces;
    try { vertices = file.request_properties_from_element("vertex", { "x", "y", "z" });} catch (const std::exception & e) {static_cast<void>(e);}
    try { normals = file.request_properties_from_element("vertex", { "nx", "ny", "nz" });} catch (const std::exception & e) {static_cast<void>(e);}
    try { texcoords = file.request_properties_from_element("vertex", { "u", "v" });} catch (const std::exception & e) {static_cast<void>(e);}
    try { faces = file.request_properties_from_element("face", { "vertex_indices" }, 3);} catch (const std::exception & e) {static_cast<void>(e);}

    file.read(file_stream);

    indexCount = faces ? 3 * static_cast<uint32_t>(faces->count) : 0;
    std::vector<uint32_t> indexBuffer(indexCount);
    std::vector<Vertex> vertexBuffer(vertices? vertices->count : 0, Vertex());

    if(vertices){
        for(size_t bufferIndex = 0, vertexIndex = 0; bufferIndex < vertices->buffer.size_bytes(); bufferIndex += 3 * sizeof(float), vertexIndex++){
            std::memcpy(&vertexBuffer[vertexIndex].pos, &vertices->buffer.get()[bufferIndex], 3 * sizeof(float));
        }
        for(uint32_t i = 0; i < vertexBuffer.size(); i++){
            maxSize = vector<float,3>(
                std::max(maxSize[0],std::abs(vertexBuffer[i].pos[0])),
                std::max(maxSize[1],std::abs(vertexBuffer[i].pos[1])),
                std::max(maxSize[2],std::abs(vertexBuffer[i].pos[2]))
                );
            bb.max = vector<float,3>(
                std::max(bb.max[0],vertexBuffer[i].pos[0]),
                std::max(bb.max[1],vertexBuffer[i].pos[1]),
                std::max(bb.max[2],vertexBuffer[i].pos[2])
            );
            bb.min = vector<float,3>(
                std::min(bb.min[0],vertexBuffer[i].pos[0]),
                std::min(bb.min[1],vertexBuffer[i].pos[1]),
                std::min(bb.min[2],vertexBuffer[i].pos[2])
            );
        }
    }
    if(faces){
        for(size_t bufferIndex = 0, index = 0; bufferIndex < faces->buffer.size_bytes(); bufferIndex += sizeof(uint32_t), index++){
            std::memcpy(&indexBuffer[index], &faces->buffer.get()[bufferIndex], sizeof(uint32_t));
        }
    }
    if(normals){
        for(size_t bufferIndex = 0, vertexIndex = 0; bufferIndex < normals->buffer.size_bytes(); bufferIndex += 3 * sizeof(float), vertexIndex++){
            std::memcpy(&vertexBuffer[vertexIndex].normal, &normals->buffer.get()[bufferIndex], 3 * sizeof(float));
        }
    } else if(vertices) {
        for(uint32_t i = 0; i < indexBuffer.size(); i += 3){
            const vector<float, 3> n = normalize(cross(
                vertexBuffer[indexBuffer[i + 1]].pos - vertexBuffer[indexBuffer[i + 0]].pos,
                vertexBuffer[indexBuffer[i + 2]].pos - vertexBuffer[indexBuffer[i + 1]].pos
            ));

            vertexBuffer[indexBuffer[i + 0]].normal += n;
            vertexBuffer[indexBuffer[i + 1]].normal += n;
            vertexBuffer[indexBuffer[i + 2]].normal += n;
        }
        for(uint32_t i = 0; i < vertexBuffer.size(); i++){
            vertexBuffer[i].normal = normalize(vertexBuffer[i].normal);
        }
    }
    if(texcoords){
        for(size_t bufferIndex = 0, vertexIndex = 0; bufferIndex < texcoords->buffer.size_bytes(); bufferIndex += 2 * sizeof(float), vertexIndex++){
            std::memcpy(&vertexBuffer[vertexIndex].uv0, &texcoords->buffer.get()[bufferIndex], 2 * sizeof(float));
        }
    }

    createBuffer(physicalDevice, device, commandBuffer, vertexBuffer.size() * sizeof(Vertex), vertexBuffer.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexStaging, this->vertices);
    createBuffer(physicalDevice, device, commandBuffer, indexBuffer.size() * sizeof(uint32_t), indexBuffer.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexStaging, indices);

    this->uniformBlock.mat = matrix<float,4,4>(1.0f);
    Buffer::create( physicalDevice,
                    device,
                    sizeof(uniformBlock),
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    &uniformBuffer.instance,
                    &uniformBuffer.memory);
    vkMapMemory(device, uniformBuffer.memory, 0, sizeof(uniformBlock), 0, &uniformBuffer.map);
    std::memcpy(uniformBuffer.map, &uniformBlock, sizeof(uniformBlock));

    Memory::nameMemory(uniformBuffer.memory, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", plyModel::loadFromFile, uniformBuffer");
}

void plyModel::createDescriptorPool(VkDevice device) {
    uint32_t imageSamplerCount = 5;
    uint32_t meshCount = 1;

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

void plyModel::createDescriptorSet(VkDevice device, texture* emptyTexture) {
    model::createMaterialDescriptorSetLayout(device, &materialDescriptorSetLayout);
    model::createNodeDescriptorSetLayout(device, &nodeDescriptorSetLayout);

    {
        VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
            descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            descriptorSetAllocInfo.descriptorPool = descriptorPool;
            descriptorSetAllocInfo.pSetLayouts = &nodeDescriptorSetLayout;
            descriptorSetAllocInfo.descriptorSetCount = 1;
        vkAllocateDescriptorSets(device, &descriptorSetAllocInfo, &uniformBuffer.descriptorSet);

        VkDescriptorBufferInfo bufferInfo{ uniformBuffer.instance, 0, sizeof(uniformBlock)};

        VkWriteDescriptorSet writeDescriptorSet{};
            writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writeDescriptorSet.descriptorCount = 1;
            writeDescriptorSet.dstSet = uniformBuffer.descriptorSet;
            writeDescriptorSet.dstBinding = 0;
            writeDescriptorSet.pBufferInfo = &bufferInfo;
        vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
    }

    {
        VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
            descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            descriptorSetAllocInfo.descriptorPool = descriptorPool;
            descriptorSetAllocInfo.pSetLayouts = &materialDescriptorSetLayout;
            descriptorSetAllocInfo.descriptorSetCount = 1;
        vkAllocateDescriptorSets(device, &descriptorSetAllocInfo, &material.descriptorSet);

        auto getDescriptorImageInfo = [&emptyTexture](texture* tex){
            VkDescriptorImageInfo descriptorImageInfo{};
            descriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            descriptorImageInfo.imageView   = tex ? *tex->getTextureImageView() : *emptyTexture->getTextureImageView();
            descriptorImageInfo.sampler     = tex ? *tex->getTextureSampler()   : *emptyTexture->getTextureSampler();
            return descriptorImageInfo;
        };

        VkDescriptorImageInfo baseColorTextureInfo{};
        if (material.pbrWorkflows.metallicRoughness){
            baseColorTextureInfo = getDescriptorImageInfo(material.baseColorTexture);
        }
        if(material.pbrWorkflows.specularGlossiness){
            baseColorTextureInfo = getDescriptorImageInfo(material.extension.diffuseTexture);
        }

        VkDescriptorImageInfo metallicRoughnessTextureInfo{};
        if (material.pbrWorkflows.metallicRoughness){
            metallicRoughnessTextureInfo = getDescriptorImageInfo(material.metallicRoughnessTexture);
        }
        if (material.pbrWorkflows.specularGlossiness){
            metallicRoughnessTextureInfo = getDescriptorImageInfo(material.extension.specularGlossinessTexture);
        }

        std::vector<VkDescriptorImageInfo> descriptorImageInfos = {
            baseColorTextureInfo,
            metallicRoughnessTextureInfo,
            getDescriptorImageInfo(material.normalTexture),
            getDescriptorImageInfo(material.occlusionTexture),
            getDescriptorImageInfo(material.emissiveTexture)
        };

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        for(const auto& info: descriptorImageInfos){
            descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = material.descriptorSet;
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size()) - 1;
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &info;
        }
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void plyModel::render(uint32_t frameIndex, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t& primitiveCount, uint32_t pushConstantSize, uint32_t pushConstantOffset, void* pushConstant) {
    static_cast<void>(frameIndex);

    std::vector<VkDescriptorSet> nodeDescriptorSets(descriptorSetsCount);
    std::copy(descriptorSets, descriptorSets + descriptorSetsCount, nodeDescriptorSets.data());
    nodeDescriptorSets.push_back(uniformBuffer.descriptorSet);
    nodeDescriptorSets.push_back(material.descriptorSet);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, descriptorSetsCount+2, nodeDescriptorSets.data(), 0, NULL);

    materialBlock.primitive = primitiveCount++;
    std::memcpy(reinterpret_cast<char*>(pushConstant) + pushConstantOffset, &materialBlock, sizeof(MaterialBlock));

    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_ALL, 0, pushConstantSize, pushConstant);

    vkCmdDrawIndexed(commandBuffer, indexCount, 1, 0, 0, 0);
}

void plyModel::renderBB(uint32_t, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t&, uint32_t pushConstantSize, uint32_t pushConstantOffset, void* pushConstant) {
    std::vector<VkDescriptorSet> nodeDescriptorSets(descriptorSetsCount);
    std::copy(descriptorSets, descriptorSets + descriptorSetsCount, nodeDescriptorSets.data());
    nodeDescriptorSets.push_back(uniformBuffer.descriptorSet);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, descriptorSetsCount + 1, nodeDescriptorSets.data(), 0, NULL);

    struct {
        alignas(16) vector<float,3> min;
        alignas(16) vector<float,3> max;
    }BB;
    BB.min = bb.min;
    BB.max = bb.max;

    std::memcpy(reinterpret_cast<char*>(pushConstant) + pushConstantOffset, &BB, sizeof(BB));

    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_ALL, 0, pushConstantSize, pushConstant);

    vkCmdDraw(commandBuffer, 36, 1, 0, 0);
}
