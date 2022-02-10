#include "object.h"
#include "core/operations.h"

#include "gltfmodel.h"

object::object(VkApplication *app) : app(app)
{
    modelMatrix = glm::mat4x4(1.0f);
    m_globalTransform = glm::mat4x4(1.0f);
    m_translate = glm::vec3(0.0f,0.0f,0.0f);
    m_rotate = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_scale = glm::vec3(1.0f,1.0f,1.0f);

    model = nullptr;
}

object::object(VkApplication *app, objectInfo info): app(app)
{
    modelMatrix = glm::mat4x4(1.0f);
    m_globalTransform = glm::mat4x4(1.0f);
    m_translate = glm::vec3(0.0f,0.0f,0.0f);
    m_rotate = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_scale = glm::vec3(1.0f,1.0f,1.0f);

    model = info.model;
    emptyTexture = info.emptyTexture;
}

object::object(VkApplication *app, gltfModel* model3D) : app(app), model(model3D)
{
    modelMatrix = glm::mat4x4(1.0f);
    m_globalTransform = glm::mat4x4(1.0f);
    m_translate = glm::vec3(0.0f,0.0f,0.0f);
    m_rotate = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_scale = glm::vec3(1.0f,1.0f,1.0f);
}

object::~object()
{

}

void object::destroyUniformBuffers()
{
    for(size_t i=0;i<uniformBuffers.size();i++)
    {
        if (uniformBuffers.at(i) != VK_NULL_HANDLE)
        {
            vkDestroyBuffer(app->getDevice(), uniformBuffers.at(i), nullptr);
            vkFreeMemory(app->getDevice(), uniformBuffersMemory.at(i), nullptr);
        }
    }
}

void object::destroyDescriptorPools()
{
    vkDestroyDescriptorPool(app->getDevice(), descriptorPool, nullptr);
}

void object::setGlobalTransform(const glm::mat4x4 & transform)
{
    m_globalTransform = transform;
    updateModelMatrix();
}

void object::translate(const glm::vec3 & translate)
{
    m_translate += translate;
    updateModelMatrix();
}

void object::rotate(const float & ang ,const glm::vec3 & ax)
{
    m_rotate = glm::quat(glm::cos(ang/2.0f),glm::sin(ang/2.0f)*glm::vec3(ax))*m_rotate;
    updateModelMatrix();
}

void object::scale(const glm::vec3 & scale)
{
    m_scale = scale;
    updateModelMatrix();
}

void object::updateModelMatrix()
{
    glm::mat4x4 translateMatrix = glm::translate(glm::mat4x4(1.0f),m_translate);
    glm::mat4x4 rotateMatrix = glm::mat4x4(1.0f);
    if(!(m_rotate.x==0&&m_rotate.y==0&&m_rotate.z==0))
    {
        rotateMatrix = glm::rotate(glm::mat4x4(1.0f),2.0f*glm::acos(m_rotate.w),glm::vec3(m_rotate.x,m_rotate.y,m_rotate.z));
    }
    glm::mat4x4 scaleMatrix = glm::scale(glm::mat4x4(1.0f),m_scale);

    modelMatrix = m_globalTransform * translateMatrix * rotateMatrix * scaleMatrix;
}

void object::updateAnimation()
{
    if(getModel()->animations.size() > 0){
        if(!changeAnimationFlag){
            if (animationTimer > getModel()->animations[animationIndex].end)
                animationTimer -= getModel()->animations[animationIndex].end;

            getModel()->updateAnimation(animationIndex, animationTimer);
        }else{
            getModel()->changeAnimation(animationIndex, newAnimationIndex, startTimer, animationTimer, changeAnimationTime);
            if(startTimer+changeAnimationTime<animationTimer){
                changeAnimationFlag = false;
                animationTimer = getModel()->animations[animationIndex+1].start;
                animationIndex = newAnimationIndex;
            }
        }
    }
}

void object::createUniformBuffers(uint32_t imageCount)
{
    uniformBuffers.resize(imageCount);
    uniformBuffersMemory.resize(imageCount);
    for (size_t i = 0; i < imageCount; i++)
        createBuffer(app,sizeof(UniformBuffer), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
}

void object::updateUniformBuffer(uint32_t currentImage)
{
    void* data;
    UniformBuffer ubo{};
        ubo.modelMatrix = modelMatrix;
        ubo.color = color;
    vkMapMemory(app->getDevice(), uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
        memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(app->getDevice(), uniformBuffersMemory[currentImage]);
}

void object::setDescriptorSetLayouts(descriptorSetLayouts setLayouts)
{
    uniformBufferSetLayout = setLayouts.uniformBufferSetLayout;
    uniformBlockSetLayout = setLayouts.uniformBlockSetLayout;
    materialSetLayout = setLayouts.materialSetLayout;
}

void object::createDescriptorPool(uint32_t imageCount)
{
    uint32_t imageSamplerCount = 0;
    uint32_t materialCount = 0;
    uint32_t meshCount = 0;
    for (auto &material : model->materials)
    {
        static_cast<void>(material);
        imageSamplerCount += 5;
        materialCount++;
    }
    for (auto node : model->linearNodes)
    {
        if (node->mesh)
        {
            meshCount++;
        }
    }

    std::vector<VkDescriptorPoolSize> DescriptorPoolSizes(3);

    size_t index = 0;

    DescriptorPoolSizes.at(index).type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    DescriptorPoolSizes.at(index).descriptorCount = static_cast<uint32_t>(imageCount);
    index++;

    DescriptorPoolSizes.at(index).type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    DescriptorPoolSizes.at(index).descriptorCount = meshCount*static_cast<uint32_t>(imageCount);
    index++;

    DescriptorPoolSizes.at(index).type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    DescriptorPoolSizes.at(index).descriptorCount = imageSamplerCount*static_cast<uint32_t>(imageCount);
    index++;

    //Мы будем выделять один из этих дескрипторов для каждого кадра. На эту структуру размера пула ссылается главный VkDescriptorPoolCreateInfo:
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(DescriptorPoolSizes.size());
    poolInfo.pPoolSizes = DescriptorPoolSizes.data();
    poolInfo.maxSets = (1+meshCount+imageSamplerCount)*static_cast<uint32_t>(imageCount);

    if (vkCreateDescriptorPool(app->getDevice(), &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create descriptor pool!");
    }
}

void object::createDescriptorSet(uint32_t imageCount)
{
    std::vector<VkDescriptorSetLayout> layouts(imageCount, *uniformBufferSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        allocInfo.pSetLayouts = layouts.data();
    descriptors.resize(imageCount);
    if (vkAllocateDescriptorSets(app->getDevice(), &allocInfo, descriptors.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate descriptor sets!");

    for (size_t i = 0; i < imageCount; i++)
    {
        VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBuffer);
        VkWriteDescriptorSet writeDescriptorSet{};
            writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writeDescriptorSet.descriptorCount = 1;
            writeDescriptorSet.dstSet = descriptors[i];
            writeDescriptorSet.dstBinding = 0;
            writeDescriptorSet.pBufferInfo = &bufferInfo;
        vkUpdateDescriptorSets(app->getDevice(), 1, &writeDescriptorSet, 0, nullptr);
    }

    for (auto node : model->linearNodes)
        if (node->mesh)
            createNodeDescriptorSet(node);

    for (auto &material : model->materials)
        createMaterialDescriptorSet(&material);
}

void object::createNodeDescriptorSet(Node* node)
{
    if (node->mesh)
    {
        VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
            descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            descriptorSetAllocInfo.descriptorPool = descriptorPool;
            descriptorSetAllocInfo.pSetLayouts = uniformBlockSetLayout;
            descriptorSetAllocInfo.descriptorSetCount = 1;
        if (vkAllocateDescriptorSets(app->getDevice(), &descriptorSetAllocInfo, &node->mesh->uniformBuffer.descriptorSet) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate descriptor sets!");

        VkWriteDescriptorSet writeDescriptorSet{};
            writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writeDescriptorSet.descriptorCount = 1;
            writeDescriptorSet.dstSet = node->mesh->uniformBuffer.descriptorSet;
            writeDescriptorSet.dstBinding = 0;
            writeDescriptorSet.pBufferInfo = &node->mesh->uniformBuffer.descriptor;
        vkUpdateDescriptorSets(app->getDevice(), 1, &writeDescriptorSet, 0, nullptr);
    }
    for (auto& child : node->children)
        createNodeDescriptorSet(child);
}

void object::createMaterialDescriptorSet(Material* material)
{
    std::vector<VkDescriptorSetLayout> layouts(1, *materialSetLayout);
    VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
    descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocInfo.descriptorPool = descriptorPool;
    descriptorSetAllocInfo.pSetLayouts = layouts.data();
    descriptorSetAllocInfo.descriptorSetCount = 1;

    if (vkAllocateDescriptorSets(app->getDevice(), &descriptorSetAllocInfo, &material->descriptorSet) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    VkDescriptorImageInfo baseColorTextureInfo;
    baseColorTextureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (material->pbrWorkflows.metallicRoughness)
    {
        baseColorTextureInfo.imageView   = material->baseColorTexture ? material->baseColorTexture->getTextureImageView() : emptyTexture->getTextureImageView();
        baseColorTextureInfo.sampler     = material->baseColorTexture ? material->baseColorTexture->getTextureSampler()   : emptyTexture->getTextureSampler();
    }
    if(material->pbrWorkflows.specularGlossiness)
    {
        baseColorTextureInfo.imageView   = material->extension.diffuseTexture ? material->extension.diffuseTexture->getTextureImageView() : emptyTexture->getTextureImageView();
        baseColorTextureInfo.sampler     = material->extension.diffuseTexture ? material->extension.diffuseTexture->getTextureSampler() : emptyTexture->getTextureSampler();
    }

    VkDescriptorImageInfo metallicRoughnessTextureInfo;
    metallicRoughnessTextureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (material->pbrWorkflows.metallicRoughness)
    {
        metallicRoughnessTextureInfo.imageView   = material->metallicRoughnessTexture ? material->metallicRoughnessTexture->getTextureImageView() : emptyTexture->getTextureImageView();
        metallicRoughnessTextureInfo.sampler     = material->metallicRoughnessTexture ? material->metallicRoughnessTexture->getTextureSampler() : emptyTexture->getTextureSampler();
    }
    if (material->pbrWorkflows.specularGlossiness)
    {
        metallicRoughnessTextureInfo.imageView   = material->extension.specularGlossinessTexture ? material->extension.specularGlossinessTexture->getTextureImageView() : emptyTexture->getTextureImageView();
        metallicRoughnessTextureInfo.sampler     = material->extension.specularGlossinessTexture ? material->extension.specularGlossinessTexture->getTextureSampler() : emptyTexture->getTextureSampler();
    }

    VkDescriptorImageInfo normalTextureInfo;
    normalTextureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    normalTextureInfo.imageView   = material->normalTexture ? material->normalTexture->getTextureImageView() : emptyTexture->getTextureImageView();
    normalTextureInfo.sampler     = material->normalTexture ? material->normalTexture->getTextureSampler() : emptyTexture->getTextureSampler();

    VkDescriptorImageInfo occlusionTextureInfo;
    occlusionTextureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    occlusionTextureInfo.imageView   = material->occlusionTexture ? material->occlusionTexture->getTextureImageView() : emptyTexture->getTextureImageView();
    occlusionTextureInfo.sampler     = material->occlusionTexture ? material->occlusionTexture->getTextureSampler() : emptyTexture->getTextureSampler();

    VkDescriptorImageInfo emissiveTextureInfo;
    emissiveTextureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    emissiveTextureInfo.imageView   = material->emissiveTexture ? material->emissiveTexture->getTextureImageView() : emptyTexture->getTextureImageView();
    emissiveTextureInfo.sampler     = material->emissiveTexture ? material->emissiveTexture->getTextureSampler() : emptyTexture->getTextureSampler();

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

    vkUpdateDescriptorSets(app->getDevice(), static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
}

void                            object::setVisibilityDistance(float visibilityDistance){this->visibilityDistance=visibilityDistance;}
void                            object::setColor(const glm::vec4 &color){this->color = color;}
void                            object::setEmptyTexture(texture* emptyTexture){this->emptyTexture = emptyTexture;}

gltfModel*                      object::getModel(){return model;}
float                           object::getVisibilityDistance(){return visibilityDistance;}
glm::vec4                       object::getColor(){return color;}

glm::mat4x4                     object::getTransformation(){return modelMatrix;}

VkDescriptorPool                &object::getDescriptorPool(){return descriptorPool;}
std::vector<VkDescriptorSet>    &object::getDescriptorSet(){return descriptors;}
