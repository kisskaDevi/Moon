#include "workflow.h"
#include "operations.h"

namespace moon::workflows {

void Workbody::destroy(VkDevice device){
    if(DescriptorPool)      {vkDestroyDescriptorPool(device, DescriptorPool, nullptr); DescriptorPool = VK_NULL_HANDLE;}
    DescriptorSets.clear();
}

void Workflow::destroy(){
    for(auto& framebuffer: framebuffers){
        if(framebuffer) vkDestroyFramebuffer(device, framebuffer,nullptr);
    }
    framebuffers.clear();
}

Workflow& Workflow::setShadersPath(const std::filesystem::path &path){
    shadersPath = path;
    return *this;
}
Workflow& Workflow::setDeviceProp(VkPhysicalDevice physicalDevice, VkDevice device){
    this->physicalDevice = physicalDevice;
    this->device = device;
    return *this;
}
Workflow& Workflow::setImageProp(moon::utils::ImageInfo* pInfo){
    this->image = *pInfo;
    return *this;
}

void Workflow::createCommandBuffers(VkCommandPool commandPool)
{
    commandBuffers.resize(image.Count);
    VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = static_cast<uint32_t>(image.Count);
    CHECK(vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()));
}

void Workflow::beginCommandBuffer(uint32_t frameNumber){
    CHECK(vkResetCommandBuffer(commandBuffers[frameNumber],0));

    VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;
        beginInfo.pInheritanceInfo = nullptr;

    CHECK(vkBeginCommandBuffer(commandBuffers[frameNumber], &beginInfo));
}

void Workflow::endCommandBuffer(uint32_t frameNumber){
    CHECK(vkEndCommandBuffer(commandBuffers[frameNumber]));
}

VkCommandBuffer& Workflow::getCommandBuffer(uint32_t frameNumber)
{
    return commandBuffers[frameNumber];
}

void Workflow::freeCommandBuffer(VkCommandPool commandPool){
    if(commandBuffers.data()){
        vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
    }
    commandBuffers.clear();
}

void Workflow::createDescriptorPool(VkDevice device, Workbody* workbody, const uint32_t& bufferCount, const uint32_t& imageCount, const uint32_t& maxSets){
    std::vector<VkDescriptorPoolSize> poolSizes;
    if(bufferCount){
        poolSizes.push_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, bufferCount});
    }
    if(imageCount){
        poolSizes.push_back(VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, imageCount});
    }
    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = maxSets;
    CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &workbody->DescriptorPool));
}

void Workflow::createDescriptorSets(VkDevice device, Workbody* workbody, const uint32_t& imageCount){
    workbody->DescriptorSets.resize(imageCount);
    std::vector<VkDescriptorSetLayout> layouts(imageCount, workbody->descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = workbody->DescriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        allocInfo.pSetLayouts = layouts.data();
    CHECK(vkAllocateDescriptorSets(device, &allocInfo, workbody->DescriptorSets.data()));
};

}
