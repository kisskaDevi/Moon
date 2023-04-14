#include "filtergraphics.h"

void filter::destroy(VkDevice device){
    if(Pipeline)            {vkDestroyPipeline(device, Pipeline, nullptr); Pipeline = VK_NULL_HANDLE;}
    if(PipelineLayout)      {vkDestroyPipelineLayout(device, PipelineLayout,nullptr); PipelineLayout = VK_NULL_HANDLE;}
    if(DescriptorSetLayout) {vkDestroyDescriptorSetLayout(device, DescriptorSetLayout, nullptr); DescriptorSetLayout = VK_NULL_HANDLE;}
    if(DescriptorPool)      {vkDestroyDescriptorPool(device, DescriptorPool, nullptr); DescriptorPool = VK_NULL_HANDLE;}
    DescriptorSets.clear();
}

void filterGraphics::destroy(){
    if(renderPass) {vkDestroyRenderPass(device, renderPass, nullptr); renderPass = VK_NULL_HANDLE;}
    for(auto& framebuffer: framebuffers){
        if(framebuffer) vkDestroyFramebuffer(device, framebuffer,nullptr);
    }
    framebuffers.clear();
}

void filterGraphics::setEmptyTexture(texture* emptyTexture){
    this->emptyTexture = emptyTexture;
}
void filterGraphics::setExternalPath(const std::string &path){
    externalPath = path;
}
void filterGraphics::setDeviceProp(VkPhysicalDevice physicalDevice, VkDevice device){
    this->physicalDevice = physicalDevice;
    this->device = device;
}
void filterGraphics::setImageProp(imageInfo* pInfo){
    this->image = *pInfo;
}
void filterGraphics::setAttachments(uint32_t attachmentsCount, attachments* pAttachments){
    this->attachmentsCount = attachmentsCount;
    this->pAttachments = pAttachments;
}

void filterGraphics::createCommandBuffers(VkCommandPool commandPool)
{
    commandBuffers.resize(image.Count);
    VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = static_cast<uint32_t>(image.Count);
    vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data());
}

void filterGraphics::beginCommandBuffer(uint32_t frameNumber){
    vkResetCommandBuffer(commandBuffers[frameNumber],0);

    VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;
        beginInfo.pInheritanceInfo = nullptr;

    vkBeginCommandBuffer(commandBuffers[frameNumber], &beginInfo);
}

void filterGraphics::endCommandBuffer(uint32_t frameNumber){
    vkEndCommandBuffer(commandBuffers[frameNumber]);
}

VkCommandBuffer& filterGraphics::getCommandBuffer(uint32_t frameNumber)
{
    return commandBuffers[frameNumber];
}

void filterGraphics::freeCommandBuffer(VkCommandPool commandPool){
    if(commandBuffers.data()){
        vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
    }
    commandBuffers.resize(0);
}

void filterGraphics::createDescriptorPool(VkDevice device, filter* filter, const uint32_t& bufferCount, const uint32_t& imageCount, const uint32_t& maxSets){
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
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &filter->DescriptorPool);
}

void filterGraphics::createDescriptorSets(VkDevice device, filter* filter, const uint32_t& imageCount){
    filter->DescriptorSets.resize(imageCount);
    std::vector<VkDescriptorSetLayout> layouts(imageCount, filter->DescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = filter->DescriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        allocInfo.pSetLayouts = layouts.data();
    vkAllocateDescriptorSets(device, &allocInfo, filter->DescriptorSets.data());
};
