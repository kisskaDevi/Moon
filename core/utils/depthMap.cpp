#include "depthMap.h"
#include "vkdefault.h"
#include "texture.h"
#include "operations.h"

namespace moon::utils {

void DepthMap::createDescriptorPool(VkDevice device, uint32_t imageCount){
    std::vector<VkDescriptorPoolSize> poolSize = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(imageCount)}
    };
    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSize.size());
        poolInfo.pPoolSizes = poolSize.data();
        poolInfo.maxSets = static_cast<uint32_t>(imageCount);
    CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool));
}

void DepthMap::createDescriptorSets(VkDevice device, uint32_t imageCount){
    createDescriptorSetLayout(device,&descriptorSetLayout);

    descriptorSets.resize(imageCount);
    std::vector<VkDescriptorSetLayout> shadowLayouts(imageCount, descriptorSetLayout);
    VkDescriptorSetAllocateInfo shadowAllocInfo{};
        shadowAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        shadowAllocInfo.descriptorPool = descriptorPool;
        shadowAllocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        shadowAllocInfo.pSetLayouts = shadowLayouts.data();
    CHECK(vkAllocateDescriptorSets(device, &shadowAllocInfo, descriptorSets.data()));
}

DepthMap::DepthMap(PhysicalDevice device, VkCommandPool commandPool, uint32_t imageCount){
    emptyTextureBlack = createEmptyTexture(device, commandPool);
    emptyTextureWhite = createEmptyTexture(device, commandPool, false);
    this->device = device.getLogical();

    createDescriptorPool(device.getLogical(), imageCount);
    createDescriptorSets(device.getLogical(), imageCount);
    updateDescriptorSets(device.getLogical(), imageCount);
}

DepthMap::~DepthMap(){
    destroy(device);
}
void DepthMap::destroy(VkDevice device){
    if(descriptorSetLayout) {vkDestroyDescriptorSetLayout(device, descriptorSetLayout,  nullptr); descriptorSetLayout = VK_NULL_HANDLE;}
    if(descriptorPool) {vkDestroyDescriptorPool(device, descriptorPool, nullptr); descriptorPool = VK_NULL_HANDLE;}

    if(emptyTextureBlack){
        emptyTextureBlack->destroy(device);
        delete emptyTextureBlack;
        emptyTextureBlack = nullptr;
    }
    if(emptyTextureWhite){
        emptyTextureWhite->destroy(device);
        delete emptyTextureWhite;
        emptyTextureWhite = nullptr;
    }
}

const std::vector<VkDescriptorSet>& DepthMap::getDescriptorSets() const{
    return descriptorSets;
}

void DepthMap::updateDescriptorSets(VkDevice device, uint32_t imageCount){
    for (size_t i = 0; i < imageCount; i++)
    {
        VkDescriptorImageInfo shadowImageInfo{};
            shadowImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            shadowImageInfo.imageView = map ? map->instances[i].imageView : *emptyTextureWhite->getTextureImageView();
            shadowImageInfo.sampler = map ? map->sampler : *emptyTextureWhite->getTextureSampler();
        std::vector<VkWriteDescriptorSet> descriptorWrites;
            descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = descriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &shadowImageInfo;
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

Attachments* &DepthMap::get(){
    return map;
}

void DepthMap::createDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout* descriptorSetLayout){
    std::vector<VkDescriptorSetLayoutBinding> binding;
        binding.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(binding.size());
        layoutInfo.pBindings = binding.data();
    CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, descriptorSetLayout));
}

}
