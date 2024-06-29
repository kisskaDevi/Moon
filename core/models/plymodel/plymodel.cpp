#include "plymodel.h"

#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"
#include "operations.h"
#include "device.h"

#include <memory>
#include <fstream>
#include <vector>

namespace moon::models {

PlyModel::PlyModel(
        std::filesystem::path filename,
        moon::math::Vector<float, 4> baseColorFactor,
        moon::math::Vector<float, 4> diffuseFactor,
        moon::math::Vector<float, 4> specularFactor,
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

moon::interfaces::MaterialBlock &PlyModel::getMaterialBlock(){
    return materialBlock;
}

PlyModel::~PlyModel() {
    PlyModel::destroy();
}

void PlyModel::destroy() {
    if(descriptorPool != VK_NULL_HANDLE){
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        descriptorPool = VK_NULL_HANDLE;
    }

    device = VK_NULL_HANDLE;
}

void PlyModel::destroyCache() {
    vertexCache = utils::Buffer();
    indexCache = utils::Buffer();
}

const VkBuffer* PlyModel::getVertices() const {
    return vertices;
}
const VkBuffer* PlyModel::getIndices() const {
    return indices;
}
const moon::math::Vector<float,3> PlyModel::getMaxSize() const {
    return maxSize;
}

void PlyModel::loadFromFile(VkPhysicalDevice physicalDevice, VkDevice device, VkCommandBuffer commandBuffer) {
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
            std::memcpy((void*)&vertexBuffer[vertexIndex].pos, (void*)&vertices->buffer.get()[bufferIndex], 3 * sizeof(float));
        }
        for(uint32_t i = 0; i < vertexBuffer.size(); i++){
            maxSize = moon::math::Vector<float,3>(
                std::max(maxSize[0],std::abs(vertexBuffer[i].pos[0])),
                std::max(maxSize[1],std::abs(vertexBuffer[i].pos[1])),
                std::max(maxSize[2],std::abs(vertexBuffer[i].pos[2]))
                );
            bb.max = moon::math::Vector<float,3>(
                std::max(bb.max[0],vertexBuffer[i].pos[0]),
                std::max(bb.max[1],vertexBuffer[i].pos[1]),
                std::max(bb.max[2],vertexBuffer[i].pos[2])
            );
            bb.min = moon::math::Vector<float,3>(
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
            std::memcpy((void*)&vertexBuffer[vertexIndex].normal, (void*)&normals->buffer.get()[bufferIndex], 3 * sizeof(float));
        }
    } else if(vertices) {
        for(uint32_t i = 0; i < indexBuffer.size(); i += 3){
            const moon::math::Vector<float, 3> n = normalize(cross(
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
            std::memcpy((void*)&vertexBuffer[vertexIndex].uv0, (void*)&texcoords->buffer.get()[bufferIndex], 2 * sizeof(float));
        }
    }

    utils::createModelBuffer(physicalDevice, device, commandBuffer, vertexBuffer.size() * sizeof(Vertex), vertexBuffer.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexCache, this->vertices);
    utils::createModelBuffer(physicalDevice, device, commandBuffer, indexBuffer.size() * sizeof(uint32_t), indexBuffer.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexCache, indices);

    this->uniformBlock.mat = moon::math::Matrix<float,4,4>(1.0f);
    uniformBuffer.create(physicalDevice, device, sizeof(uniformBlock), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    uniformBuffer.copy(&uniformBlock);
    moon::utils::Memory::instance().nameMemory(uniformBuffer, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", plyModel::loadFromFile, uniformBuffer");
}

void PlyModel::createDescriptorPool() {
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
    CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool));
}

void PlyModel::createDescriptorSet() {
    nodeDescriptorSetLayout = moon::interfaces::Model::createNodeDescriptorSetLayout(device);
    materialDescriptorSetLayout = moon::interfaces::Model::createMaterialDescriptorSetLayout(device);

    {
        VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
            descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            descriptorSetAllocInfo.descriptorPool = descriptorPool;
            descriptorSetAllocInfo.pSetLayouts = nodeDescriptorSetLayout;
            descriptorSetAllocInfo.descriptorSetCount = 1;
        CHECK(vkAllocateDescriptorSets(device, &descriptorSetAllocInfo, &uniformBuffer.descriptorSet));

        VkDescriptorBufferInfo bufferInfo{ uniformBuffer, 0, sizeof(uniformBlock)};

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
            descriptorSetAllocInfo.pSetLayouts = materialDescriptorSetLayout;
            descriptorSetAllocInfo.descriptorSetCount = 1;
        CHECK(vkAllocateDescriptorSets(device, &descriptorSetAllocInfo, &material.descriptorSet));

        auto getDescriptorImageInfo = [this](const moon::utils::Texture* tex){
            VkDescriptorImageInfo descriptorImageInfo{};
            descriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            descriptorImageInfo.imageView   = tex ? tex->imageView() : emptyTexture.imageView();
            descriptorImageInfo.sampler     = tex ? tex->sampler()   : emptyTexture.sampler();
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

void PlyModel::create(const moon::utils::PhysicalDevice& device, VkCommandPool commandPool)
{
    if(this->device)
    {
        CHECK_M(commandPool == VK_NULL_HANDLE, std::string("[ deferredGraphics::createModel ] VkCommandPool is VK_NULL_HANDLE"));
        CHECK_M(device.instance == VK_NULL_HANDLE, std::string("[ deferredGraphics::createModel ] VkPhysicalDevice is VK_NULL_HANDLE"));
        CHECK_M(device.getLogical() == VK_NULL_HANDLE, std::string("[ deferredGraphics::createModel ] VkDevice is VK_NULL_HANDLE"));

        emptyTexture = utils::Texture::empty(device, commandPool);
        this->device = device.getLogical();

        VkCommandBuffer commandBuffer = moon::utils::singleCommandBuffer::create(device.getLogical(),commandPool);
        loadFromFile(device.instance, device.getLogical(), commandBuffer);
        moon::utils::singleCommandBuffer::submit(device.getLogical(),device.getQueue(0,0), commandPool, &commandBuffer);
        destroyCache();
        createDescriptorPool();
        createDescriptorSet();
    }
}

void PlyModel::render(uint32_t frameIndex, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t& primitiveCount, uint32_t pushConstantSize, uint32_t pushConstantOffset, void* pushConstant) {
    static_cast<void>(frameIndex);

    std::vector<VkDescriptorSet> nodeDescriptorSets(descriptorSetsCount);
    std::copy(descriptorSets, descriptorSets + descriptorSetsCount, nodeDescriptorSets.data());
    nodeDescriptorSets.push_back(uniformBuffer.descriptorSet);
    nodeDescriptorSets.push_back(material.descriptorSet);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, descriptorSetsCount+2, nodeDescriptorSets.data(), 0, NULL);

    materialBlock.primitive = primitiveCount++;
    std::memcpy(reinterpret_cast<char*>(pushConstant) + pushConstantOffset, &materialBlock, sizeof(moon::interfaces::MaterialBlock));

    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_ALL, 0, pushConstantSize, pushConstant);

    vkCmdDrawIndexed(commandBuffer, indexCount, 1, 0, 0, 0);
}

void PlyModel::renderBB(uint32_t, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets) {
    std::vector<VkDescriptorSet> nodeDescriptorSets(descriptorSetsCount);
    std::copy(descriptorSets, descriptorSets + descriptorSetsCount, nodeDescriptorSets.data());
    nodeDescriptorSets.push_back(uniformBuffer.descriptorSet);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, descriptorSetsCount + 1, nodeDescriptorSets.data(), 0, NULL);

    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(moon::interfaces::BoundingBox), &bb);
    vkCmdDraw(commandBuffer, 24, 1, 0, 0);
}

}
