#include "workflow.h"

void workbody::destroy(VkDevice device){
    if(Pipeline)            {vkDestroyPipeline(device, Pipeline, nullptr); Pipeline = VK_NULL_HANDLE;}
    if(PipelineLayout)      {vkDestroyPipelineLayout(device, PipelineLayout,nullptr); PipelineLayout = VK_NULL_HANDLE;}
    if(DescriptorSetLayout) {vkDestroyDescriptorSetLayout(device, DescriptorSetLayout, nullptr); DescriptorSetLayout = VK_NULL_HANDLE;}
    if(DescriptorPool)      {vkDestroyDescriptorPool(device, DescriptorPool, nullptr); DescriptorPool = VK_NULL_HANDLE;}
    DescriptorSets.clear();
}

void workflow::destroy(){
    if(renderPass) {vkDestroyRenderPass(device, renderPass, nullptr); renderPass = VK_NULL_HANDLE;}
    for(auto& framebuffer: framebuffers){
        if(framebuffer) vkDestroyFramebuffer(device, framebuffer,nullptr);
    }
    framebuffers.clear();
}

void workflow::setEmptyTexture(texture* emptyTexture){
    this->emptyTexture = emptyTexture;
}
void workflow::setShadersPath(const std::filesystem::path &path){
    shadersPath = path;
}
void workflow::setDeviceProp(VkPhysicalDevice physicalDevice, VkDevice device){
    this->physicalDevice = physicalDevice;
    this->device = device;
}
void workflow::setImageProp(imageInfo* pInfo){
    this->image = *pInfo;
}
void workflow::setAttachments(uint32_t attachmentsCount, attachments* pAttachments){
    this->attachmentsCount = attachmentsCount;
    this->pAttachments = pAttachments;
}

void workflow::createCommandBuffers(VkCommandPool commandPool)
{
    commandBuffers.resize(image.Count);
    VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = static_cast<uint32_t>(image.Count);
    vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data());
}

void workflow::beginCommandBuffer(uint32_t frameNumber){
    vkResetCommandBuffer(commandBuffers[frameNumber],0);

    VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;
        beginInfo.pInheritanceInfo = nullptr;

    vkBeginCommandBuffer(commandBuffers[frameNumber], &beginInfo);
}

void workflow::endCommandBuffer(uint32_t frameNumber){
    vkEndCommandBuffer(commandBuffers[frameNumber]);
}

VkCommandBuffer& workflow::getCommandBuffer(uint32_t frameNumber)
{
    return commandBuffers[frameNumber];
}

void workflow::freeCommandBuffer(VkCommandPool commandPool){
    if(commandBuffers.data()){
        vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
    }
    commandBuffers.resize(0);
}

void workflow::createDescriptorPool(VkDevice device, workbody* workbody, const uint32_t& bufferCount, const uint32_t& imageCount, const uint32_t& maxSets){
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
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &workbody->DescriptorPool);
}

void workflow::createDescriptorSets(VkDevice device, workbody* workbody, const uint32_t& imageCount){
    workbody->DescriptorSets.resize(imageCount);
    std::vector<VkDescriptorSetLayout> layouts(imageCount, workbody->DescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = workbody->DescriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(imageCount);
        allocInfo.pSetLayouts = layouts.data();
    vkAllocateDescriptorSets(device, &allocInfo, workbody->DescriptorSets.data());
};
