#include "depthMap.h"
#include "vkdefault.h"
#include "operations.h"

namespace moon::utils {

void DepthMap::createDescriptorPool(uint32_t imageCount){
    descriptorSetLayout = createDescriptorSetLayout(device);
    CHECK(descriptorPool.create(device, { &descriptorSetLayout }, imageCount));
    descriptorSets = descriptorPool.allocateDescriptorSets(descriptorSetLayout, imageCount);
}

DepthMap::DepthMap(const PhysicalDevice& device, VkCommandPool commandPool, uint32_t imageCount){
    emptyTextureWhite = utils::Texture::empty(device, commandPool, false);
    this->device = device.getLogical();

    createDescriptorPool(imageCount);
    updateDescriptorSets(imageCount);
}

DepthMap::~DepthMap(){
    destroy(device);
}

void DepthMap::destroy(VkDevice device){
    if(map) {
        delete map;
        map = nullptr;
    }
}

const utils::vkDefault::DescriptorSets& DepthMap::getDescriptorSets() const{
    return descriptorSets;
}

void DepthMap::updateDescriptorSets(uint32_t imageCount){
    for (size_t i = 0; i < imageCount; i++)
    {
        VkDescriptorImageInfo shadowImageInfo{};
            shadowImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            shadowImageInfo.imageView = map ? map->imageView(i) : emptyTextureWhite.imageView();
            shadowImageInfo.sampler = map ? map->sampler() : emptyTextureWhite.sampler();
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

moon::utils::vkDefault::DescriptorSetLayout DepthMap::createDescriptorSetLayout(VkDevice device){
    moon::utils::vkDefault::DescriptorSetLayout descriptorSetLayout;

    std::vector<VkDescriptorSetLayoutBinding> binding;
        binding.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));

    CHECK(descriptorSetLayout.create(device, binding));
    return descriptorSetLayout;
}

}
