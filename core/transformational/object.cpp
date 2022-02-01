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

    m_model = nullptr;
}

object::object(VkApplication *app, objectInfo info): app(app)
{
    modelMatrix = glm::mat4x4(1.0f);
    m_globalTransform = glm::mat4x4(1.0f);
    m_translate = glm::vec3(0.0f,0.0f,0.0f);
    m_rotate = glm::quat(1.0f,0.0f,0.0f,0.0f);
    m_scale = glm::vec3(1.0f,1.0f,1.0f);

    m_model = info.model;
    m_emptyTexture = info.emptyTexture;
}

object::object(VkApplication *app, gltfModel* model3D) : app(app), m_model(model3D)
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
    for(size_t i=0;i<m_uniformBuffers.size();i++)
    {
        if (m_uniformBuffers.at(i) != VK_NULL_HANDLE)
        {
            vkDestroyBuffer(app->getDevice(), m_uniformBuffers.at(i), nullptr);
            vkFreeMemory(app->getDevice(), m_uniformBuffersMemory.at(i), nullptr);
        }
    }
}

void object::destroyDescriptorPools()
{
    vkDestroyDescriptorPool(app->getDevice(), m_descriptorPool, nullptr);
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
    VkDeviceSize bufferSize = sizeof(UniformBuffer);

    m_uniformBuffers.resize(imageCount);
    m_uniformBuffersMemory.resize(imageCount);

    for (size_t i = 0; i < imageCount; i++)
    {
        createBuffer(app,bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, m_uniformBuffers[i], m_uniformBuffersMemory[i]);
    }
}

void object::updateUniformBuffer(uint32_t currentImage)
{
    UniformBuffer ubo{};

    ubo.modelMatrix = modelMatrix;

    void* data;
    vkMapMemory(app->getDevice(), m_uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
        memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(app->getDevice(), m_uniformBuffersMemory[currentImage]);
}

void object::setDescriptorSetLayouts(descriptorSetLayouts setLayouts)
{
    m_uniformBufferSetLayout = setLayouts.uniformBufferSetLayout;
    m_uniformBlockSetLayout = setLayouts.uniformBlockSetLayout;
    m_materialSetLayout = setLayouts.materialSetLayout;
}

void object::createDescriptorPool(uint32_t imageCount)
{
    uint32_t imageSamplerCount = 0;
    uint32_t materialCount = 0;
    uint32_t meshCount = 0;
    for (auto &material : m_model->materials)
    {
        static_cast<void>(material);
        imageSamplerCount += 5;
        materialCount++;
    }
    for (auto node : m_model->linearNodes)
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

    if (vkCreateDescriptorPool(app->getDevice(), &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create descriptor pool!");
    }
}

void object::createDescriptorSet(uint32_t imageCount)
{
    std::vector<VkDescriptorSetLayout> layouts(imageCount, *m_uniformBufferSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
    allocInfo.pSetLayouts = layouts.data();

    m_descriptors.resize(imageCount);
    if (vkAllocateDescriptorSets(app->getDevice(), &allocInfo, m_descriptors.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    //Наборы дескрипторов уже выделены, но дескрипторы внутри еще нуждаются в настройке.
    //Теперь мы добавим цикл для заполнения каждого дескриптора:
    for (size_t i = 0; i < imageCount; i++)
    {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = m_uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBuffer);

        VkWriteDescriptorSet writeDescriptorSet{};
        writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeDescriptorSet.descriptorCount = 1;
        writeDescriptorSet.dstSet = m_descriptors[i];
        writeDescriptorSet.dstBinding = 0;
        writeDescriptorSet.pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(app->getDevice(), 1, &writeDescriptorSet, 0, nullptr);
    }

    for (auto node : m_model->linearNodes)
    {
        if (node->mesh)
        {
            createNodeDescriptorSet(node);
        }
    }

    for (auto &material : m_model->materials)
    {
        createMaterialDescriptorSet(&material);
    }
}

void object::createNodeDescriptorSet(Node* node)
{
    if (node->mesh)
    {
        VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
        descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocInfo.descriptorPool = m_descriptorPool;
        descriptorSetAllocInfo.pSetLayouts = m_uniformBlockSetLayout;
        descriptorSetAllocInfo.descriptorSetCount = 1;

        if (vkAllocateDescriptorSets(app->getDevice(), &descriptorSetAllocInfo, &node->mesh->uniformBuffer.descriptorSet) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

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
    {
        createNodeDescriptorSet(child);
    }
}

void object::createMaterialDescriptorSet(Material* material)
{
    std::vector<VkDescriptorSetLayout> layouts(1, *m_materialSetLayout);
    VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
    descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocInfo.descriptorPool = m_descriptorPool;
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
        baseColorTextureInfo.imageView   = material->baseColorTexture ? material->baseColorTexture->getTextureImageView() : m_emptyTexture->getTextureImageView();
        baseColorTextureInfo.sampler     = material->baseColorTexture ? material->baseColorTexture->getTextureSampler()   : m_emptyTexture->getTextureSampler();
    }
    if(material->pbrWorkflows.specularGlossiness)
    {
        baseColorTextureInfo.imageView   = material->extension.diffuseTexture ? material->extension.diffuseTexture->getTextureImageView() : m_emptyTexture->getTextureImageView();
        baseColorTextureInfo.sampler     = material->extension.diffuseTexture ? material->extension.diffuseTexture->getTextureSampler() : m_emptyTexture->getTextureSampler();
    }

    VkDescriptorImageInfo metallicRoughnessTextureInfo;
    metallicRoughnessTextureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (material->pbrWorkflows.metallicRoughness)
    {
        metallicRoughnessTextureInfo.imageView   = material->metallicRoughnessTexture ? material->metallicRoughnessTexture->getTextureImageView() : m_emptyTexture->getTextureImageView();
        metallicRoughnessTextureInfo.sampler     = material->metallicRoughnessTexture ? material->metallicRoughnessTexture->getTextureSampler() : m_emptyTexture->getTextureSampler();
    }
    if (material->pbrWorkflows.specularGlossiness)
    {
        metallicRoughnessTextureInfo.imageView   = material->extension.specularGlossinessTexture ? material->extension.specularGlossinessTexture->getTextureImageView() : m_emptyTexture->getTextureImageView();
        metallicRoughnessTextureInfo.sampler     = material->extension.specularGlossinessTexture ? material->extension.specularGlossinessTexture->getTextureSampler() : m_emptyTexture->getTextureSampler();
    }

    VkDescriptorImageInfo normalTextureInfo;
    normalTextureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    normalTextureInfo.imageView   = material->normalTexture ? material->normalTexture->getTextureImageView() : m_emptyTexture->getTextureImageView();
    normalTextureInfo.sampler     = material->normalTexture ? material->normalTexture->getTextureSampler() : m_emptyTexture->getTextureSampler();

    VkDescriptorImageInfo occlusionTextureInfo;
    occlusionTextureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    occlusionTextureInfo.imageView   = material->occlusionTexture ? material->occlusionTexture->getTextureImageView() : m_emptyTexture->getTextureImageView();
    occlusionTextureInfo.sampler     = material->occlusionTexture ? material->occlusionTexture->getTextureSampler() : m_emptyTexture->getTextureSampler();

    VkDescriptorImageInfo emissiveTextureInfo;
    emissiveTextureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    emissiveTextureInfo.imageView   = material->emissiveTexture ? material->emissiveTexture->getTextureImageView() : m_emptyTexture->getTextureImageView();
    emissiveTextureInfo.sampler     = material->emissiveTexture ? material->emissiveTexture->getTextureSampler() : m_emptyTexture->getTextureSampler();

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
void                            object::setEmptyTexture(texture* emptyTexture){m_emptyTexture = emptyTexture;}

gltfModel*                      object::getModel(){return m_model;}

float                           object::getVisibilityDistance(){return visibilityDistance;}
glm::mat4x4                     object::getTransformation(){return modelMatrix;}

VkDescriptorPool                &object::getDescriptorPool(){return m_descriptorPool;}
std::vector<VkDescriptorSet>    &object::getDescriptorSet(){return m_descriptors;}
