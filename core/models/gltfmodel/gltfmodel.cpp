#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define STBI_MSC_SECURE_CRT

#include "gltfmodel.h"
#include "operations.h"
#include "device.h"

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
            vector<float,3> dv1   = vertexBuffer[indexBuffer[i+1]].pos - vertexBuffer[indexBuffer[i+0]].pos;
            vector<float,3> dv2   = vertexBuffer[indexBuffer[i+2]].pos - vertexBuffer[indexBuffer[i+0]].pos;

            vector<float,2> duv1  = vertexBuffer[indexBuffer[i+1]].uv0 - vertexBuffer[indexBuffer[i+0]].uv0;
            vector<float,2> duv2  = vertexBuffer[indexBuffer[i+2]].uv0 - vertexBuffer[indexBuffer[i+0]].uv0;

            float det = 1.0f/(duv1[0]*duv2[1] - duv1[1]*duv2[0]);

            vector<float,3> tangent = normalize( det*(duv2[1]*dv1-duv1[1]*dv2));
            vector<float,3> bitangent = normalize( det*(duv1[0]*dv2-duv2[0]*dv1));

            if(dot(cross(tangent,bitangent),vertexBuffer[indexBuffer[i+0]].normal)<0.0f){
                tangent = -1.0f * tangent;
            }

            for(uint32_t index = 0; index < 3; index++){
                vertexBuffer[indexBuffer[i+index]].tangent      = normalize(tangent - vertexBuffer[indexBuffer[i+index]].normal * dot(vertexBuffer[indexBuffer[i+index]].normal, tangent));
                vertexBuffer[indexBuffer[i+index]].bitangent    = normalize(cross(vertexBuffer[indexBuffer[i+index]].normal, vertexBuffer[indexBuffer[i+index]].tangent));
            }
        }
    }

    bool isBinary(const std::filesystem::path& filename){
        size_t extpos = filename.string().rfind('.', filename.string().length());
        return (extpos != std::string::npos) && (filename.string().substr(extpos + 1, filename.string().length() - extpos) == "glb");
    }

    void createNodeDescriptorSet(VkDevice device, Node* node, VkDescriptorPool descriptorPool, VkDescriptorSetLayout descriptorSetLayout)
    {
        if (node->mesh){
            VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
                descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                descriptorSetAllocInfo.descriptorPool = descriptorPool;
                descriptorSetAllocInfo.pSetLayouts = &descriptorSetLayout;
                descriptorSetAllocInfo.descriptorSetCount = 1;
            CHECK(vkAllocateDescriptorSets(device, &descriptorSetAllocInfo, &node->mesh->uniformBuffer.descriptorSet));

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
        CHECK(vkAllocateDescriptorSets(device, &descriptorSetAllocInfo, &material->descriptorSet));

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
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size()) - 1;
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &info;
        }
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

gltfModel::gltfModel(std::filesystem::path filename, uint32_t instanceCount)
    : filename(filename){
    instances.resize(instanceCount);
}

gltfModel::~gltfModel() {
    gltfModel::destroy(device);
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

    if(emptyTexture){
        emptyTexture->destroy(device);
        delete emptyTexture;
        emptyTexture = nullptr;
    }

    created = false;
};

void gltfModel::destroyStagingBuffer(VkDevice device)
{
    for(auto& texture: textures){
        texture.destroyStagingBuffer(device);
    }
    for(auto texture: textureStaging){
        delete[] texture;
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
                std::memcpy((void*)newSkin->inverseBindMatrices.data(), (void*)&buffer.data[accessor.byteOffset + bufferView.byteOffset], accessor.count * sizeof(matrix<float,4,4>));
                for(auto& matrix: newSkin->inverseBindMatrices){
                    matrix = transpose(matrix);
                }
            }

            for (const auto& node: gltfModel.nodes) {
                if(node.skin == &source - &gltfModel.skins[0]){
                    nodeFromIndex(static_cast<uint32_t>(&node - &gltfModel.nodes[0]), instance.nodes)->skin = newSkin;
                }
            }

            instance.skins.push_back(newSkin);
        }
    }
}

void gltfModel::loadTextures(VkPhysicalDevice physicalDevice, VkDevice device, VkCommandBuffer commandBuffer, tinygltf::Model& gltfModel)
{
    for(const tinygltf::Texture &tex : gltfModel.textures){
        const tinygltf::Image& gltfimage = gltfModel.images[tex.source];
        const VkDeviceSize bufferSize = gltfimage.width * gltfimage.height * 4;
        if (gltfimage.component == 3){
            textureStaging.push_back(new stbi_uc[bufferSize]);
            for (int32_t i = 0; i< gltfimage.width * gltfimage.height; ++i){
                for (int32_t j = 0; j < 3; ++j) {
                    textureStaging.back()[4*i + j] = gltfimage.image[3*i + j];
                }
                textureStaging.back()[4*i + 3] = 255;
            }
        }
        const unsigned char* buffer = gltfimage.component == 3 ? textureStaging.back() : gltfimage.image.data();

        textureSampler TextureSampler{};
            TextureSampler.minFilter = tex.sampler == -1 ? VK_FILTER_LINEAR : getVkFilterMode(gltfModel.samplers[tex.sampler].minFilter);
            TextureSampler.magFilter = tex.sampler == -1 ? VK_FILTER_LINEAR : getVkFilterMode(gltfModel.samplers[tex.sampler].magFilter);
            TextureSampler.addressModeU = tex.sampler == -1 ? VK_SAMPLER_ADDRESS_MODE_REPEAT : getVkWrapMode(gltfModel.samplers[tex.sampler].wrapS);
            TextureSampler.addressModeV = tex.sampler == -1 ? VK_SAMPLER_ADDRESS_MODE_REPEAT : getVkWrapMode(gltfModel.samplers[tex.sampler].wrapT);
            TextureSampler.addressModeW = tex.sampler == -1 ? VK_SAMPLER_ADDRESS_MODE_REPEAT : TextureSampler.addressModeV;
        textures.emplace_back(texture());
        textures.back().createTextureImage(physicalDevice, device, commandBuffer, gltfimage.width, gltfimage.height, (void**) &buffer);
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
            auto color = mat.values["baseColorFactor"].ColorFactor();
            material.baseColorFactor = vector<float, 4>(
                static_cast<float>(color[0]),
                static_cast<float>(color[1]),
                static_cast<float>(color[2]),
                static_cast<float>(color[3]));
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
            auto color = mat.additionalValues["emissiveFactor"].ColorFactor();
            material.emissiveFactor = vector<float, 4>(
                static_cast<float>(color[0]),
                static_cast<float>(color[1]),
                static_cast<float>(color[2]), 1.0f);
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

    if (std::string error{}, warning{}; isBinary(filename) ?
        gltfContext.LoadBinaryFromFile(&gltfModel, &error, &warning, filename.string().c_str()) :
        gltfContext.LoadASCIIFromFile(&gltfModel, &error, &warning, filename.string().c_str()))
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
    CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool));
}

void gltfModel::createDescriptorSet(VkDevice device, texture* emptyTexture)
{
    model::createMaterialDescriptorSetLayout(device, &materialDescriptorSetLayout);
    model::createNodeDescriptorSetLayout(device, &nodeDescriptorSetLayout);

    for(auto& instance : instances){
        for (auto& node : instance.nodes){
            createNodeDescriptorSet(device, node , descriptorPool, nodeDescriptorSetLayout);
        }
    }

    for (auto &material : materials){
        createMaterialDescriptorSet(device, &material, emptyTexture, descriptorPool, materialDescriptorSetLayout);
    }
}

void gltfModel::create(physicalDevice device, VkCommandPool commandPool)
{
    if(!created)
    {
        CHECK_M(commandPool == VK_NULL_HANDLE, std::string("[ deferredGraphics::createModel ] VkCommandPool is VK_NULL_HANDLE"));
        CHECK_M(device.instance == VK_NULL_HANDLE, std::string("[ deferredGraphics::createModel ] VkPhysicalDevice is VK_NULL_HANDLE"));
        CHECK_M(device.getLogical() == VK_NULL_HANDLE, std::string("[ deferredGraphics::createModel ] VkDevice is VK_NULL_HANDLE"));

        emptyTexture = createEmptyTexture(device, commandPool);
        this->device = device.getLogical();

        VkCommandBuffer commandBuffer = SingleCommandBuffer::create(device.getLogical(),commandPool);
        loadFromFile(device.instance, device.getLogical(), commandBuffer);
        SingleCommandBuffer::submit(device.getLogical(),device.getQueue(0,0), commandPool, &commandBuffer);
        destroyStagingBuffer(device.getLogical());
        createDescriptorPool(device.getLogical());
        createDescriptorSet(device.getLogical(), emptyTexture);
        created = true;
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
                material.specularFactor = vector<float, 4>(
                    primitive->material->extension.specularFactor[0],
                    primitive->material->extension.specularFactor[1],
                    primitive->material->extension.specularFactor[2],
                    1.0f);
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

void renderNodeBB(Node *node, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t& primitiveCount, uint32_t pushConstantSize, uint32_t pushConstantOffset, void* pushConstant)
{
    if (node->mesh)
    {
        for (Mesh::Primitive* primitive : node->mesh->primitives)
        {
            if(primitive->bb.valid){
                std::vector<VkDescriptorSet> nodeDescriptorSets(descriptorSetsCount);
                std::copy(descriptorSets, descriptorSets + descriptorSetsCount, nodeDescriptorSets.data());
                nodeDescriptorSets.push_back(node->mesh->uniformBuffer.descriptorSet);
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, descriptorSetsCount + 1, nodeDescriptorSets.data(), 0, NULL);

                struct {
                    alignas(16) vector<float,3> min;
                    alignas(16) vector<float,3> max;
                }BB;
                BB.min = primitive->bb.min;
                BB.max = primitive->bb.max;

                std::memcpy(reinterpret_cast<char*>(pushConstant) + pushConstantOffset, &BB, sizeof(BB));

                vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_ALL, 0, pushConstantSize, pushConstant);

                vkCmdDraw(commandBuffer, 36, 1, 0, 0);
            }
        }
    }
    for (auto child : node->children){
        renderNodeBB(child, commandBuffer, pipelineLayout, descriptorSetsCount, descriptorSets, primitiveCount, pushConstantSize, pushConstantOffset, pushConstant);
    }
}

void gltfModel::renderBB(uint32_t frameIndex, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t &primitiveCount, uint32_t pushConstantSize, uint32_t pushConstantOffset, void* pushConstant){
    for (auto node: instances[frameIndex].nodes){
        renderNodeBB(node, commandBuffer, pipelineLayout, descriptorSetsCount, descriptorSets, primitiveCount, pushConstantSize, pushConstantOffset, pushConstant);
    }
}

